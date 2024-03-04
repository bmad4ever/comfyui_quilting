from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Event
from threading import Thread
from comfy import utils
import numpy as np
import numpy.random
import torch


# TODO add nodes where user defines output's height and width instead of scale

def quilt_single_src_no_parallelization(src, block_size, overlap, outH, outW, tolerance, rng: np.random.Generator,
                                        jobs_shm_name, job_id):
    from .quilting.generate import generateTextureMap
    return generateTextureMap(src, block_size, overlap, outH, outW, tolerance, rng, jobs_shm_name, job_id)


def quilt_single_with_parallelization(src, block_size, overlap, outH, outW, tolerance, parallelization_lvl, rng,
                                      jobs_shm_name, job_id):
    from .parallel_quilting import generateTextureMap_p
    return generateTextureMap_p(src, block_size, overlap, outH, outW, tolerance, parallelization_lvl, rng,
                                jobs_shm_name, job_id)


QUILTING_SHARED_INPUT_TYPES = {
    # block size is given in pixels
    "block_size": ("INT", {"default": 20, "min": 3, "max": 256, "step": 1}),

    # the percentage of pixels that overlap between each block_sized block
    "overlap": ("FLOAT", {"default": 1 / 6.0, "min": .1, "max": .9, "step": .01}),

    # this is a percentage relative to min error when searching for a patch.
    # the ones that within tolerance are potential candidates to be selected.
    # tolerance equal to 1 means a tolerance of 2 times the min error.
    # my interpretation regarding its application is that
    # tolerance can help prevent too much sameness in the texture
    # due to some subset of patches being better at minimizing the error.
    "tolerance": ("FLOAT", {"default": .1, "min": 0.01, "max": 2, "step": .01}),

    # 0 -> no parallelization; uses the reference implementation
    # 1 -> 4 jobs used ( divides generation into 4 sections )
    # 2 and above ->  the number of jobs per section
    # parallelization lvl also affects output; even if seed is fixed, changing this will change the output
    "parallelization_lvl": ("INT", {"default": 1, "min": 0, "max": 6, "step": 1}),

    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
}


def waiting_loop(abort_loop_event: Event, pbar: utils.ProgressBar, total_steps, shm_name, ntasks=1):
    """
        Listens for interrupts and propagates to Problem running using interruption_proxy.
    Updates progress_bar via ticker_proxy, updated within Problem instances.

    @param abort_loop_event: to be triggered in the main thread once the job's done
    @param pbar: comfyui progress bar to update every 100 milliseconds
    @param total_steps: the total number of patches to quilt
    @param shm_name: shared memory name for a numpy integer array
                        the first index is used to stop the processes in case of an interruption
                        the remaining indexes store the number of patches places by each process
    """
    from time import sleep
    from comfy.model_management import processing_interrupted
    shm = SharedMemory(name=shm_name)
    procs_statuses = np.ndarray((1 + ntasks,), dtype=np.dtype('uint32'), buffer=shm.buf)
    while not abort_loop_event.is_set():
        sleep(.1)  # pause for 1 second
        if processing_interrupted():
            procs_statuses[0] = 1
            return
        pbar.update_absolute(int(np.sum(procs_statuses[1:ntasks + 1])), total_steps)
    return


def terminate_generation(finished_event, jobs_shared_memory, pbt: Thread):
    coord_jobs_array = np.ndarray((1,), dtype=np.dtype('uint32'), buffer=jobs_shared_memory.buf)
    interrupted = coord_jobs_array[0]
    finished_event.set()
    pbt.join()
    jobs_shared_memory.close()
    jobs_shared_memory.unlink()
    if interrupted:
        from comfy.model_management import throw_exception_if_processing_interrupted
        throw_exception_if_processing_interrupted()


def setup_pbar(blocksize, overlap, out_height, out_width, par_lvl, batch_len):
    nH = int(np.ceil((out_height - blocksize) * 1.0 / (blocksize - overlap)))
    nW = int(np.ceil((out_width - blocksize) * 1.0 / (blocksize - overlap)))
    total_steps: int = ((nH + 1) * (nW + 1) - 1) * batch_len  # ignores first corner/center
    # might be less than the generated when using parallel solution, but not by far, so will leave it like this for now

    pbar: utils.ProgressBar = utils.ProgressBar(total_steps)
    finished_event = Event()

    if batch_len > 1:
        n_jobs = batch_len
        n_jobs *= 1 if par_lvl == 0 else 4
    else:
        n_jobs = 1 if par_lvl == 0 else 4
        n_jobs *= par_lvl if par_lvl > 0 else 1

    size = (1 + n_jobs) * np.dtype('uint32').itemsize
    shm_jobs = SharedMemory(create=True, size=size)

    t = Thread(target=waiting_loop, args=(finished_event, pbar, total_steps, shm_jobs.name, n_jobs))
    t.start()

    return finished_event, t, shm_jobs.name, shm_jobs


class ImageQuilting:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src": ("IMAGE",),
                # can be a single image or a batch of images
                # using an image batch limits max parallelization lvl to 1 ( values above are ignored ).

                # self explanatory
                "scale": ("FLOAT", {"default": 4, "min": 2, "max": 10, "step": .1}),

                **QUILTING_SHARED_INPUT_TYPES
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compute"
    CATEGORY = "Bmad/CV/Misc"

    def compute(self, src, block_size, scale, overlap, tolerance, parallelization_lvl, seed):
        H, W = src.shape[1:3]
        outH, outW = int(scale * H), int(scale * W)
        if overlap > 0:
            overlap = int(block_size * overlap)
        else:
            overlap = int(block_size / 6.0)

        # note: the input src should have normalized values, not 0 to 255

        rng: numpy.random.Generator = np.random.default_rng(seed=seed)

        finish_event, t, shm_name, shm_jobs = \
            setup_pbar(block_size, overlap, outH, outW, parallelization_lvl, src.shape[0])

        if src.shape[0] > 1:  # if image batch
            texture_batch = self.batch_using_jobs(src, block_size, overlap, outH, outW, tolerance,
                                                  parallelization_lvl, rng, shm_name)
            terminate_generation(finish_event, shm_jobs, t)
            return (texture_batch,)

        # if single image
        src = src.cpu().numpy().squeeze()
        if parallelization_lvl == 0:
            texture = quilt_single_src_no_parallelization(src, block_size, overlap, outH, outW, tolerance, rng,
                                                          shm_name, 0)
        else:
            texture = quilt_single_with_parallelization(
                src, block_size, overlap, outH, outW, tolerance, parallelization_lvl, rng, shm_name, 0)

        texture = torch.from_numpy(texture).unsqueeze(0)

        terminate_generation(finish_event, shm_jobs, t)
        return (texture,)

    @staticmethod
    def batch_using_jobs(src, block_size, overlap, outH, outW, tolerance, parallelization_lvl, rng, jobs_shm_name):
        from joblib import Parallel, delayed

        def unwrap_and_quilt(img_as_tensor, job_id):
            image = img_as_tensor.cpu().numpy()
            if parallelization_lvl == 0:
                result = quilt_single_src_no_parallelization(image, block_size, overlap, outH, outW, tolerance, rng,
                                                             jobs_shm_name, job_id)
            else:
                result = quilt_single_with_parallelization(image, block_size, overlap, outH, outW, tolerance, 1, rng,
                                                           jobs_shm_name, job_id)
            return torch.from_numpy(result)

        results = Parallel(n_jobs=-1, backend="loky", timeout=None)(
            delayed(unwrap_and_quilt)(src[i], i) for i in range(src.shape[0]))

        return torch.stack(results)


class LatentQuilting:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src": ("LATENT",),

                # self explanatory
                "scale": ("FLOAT", {"default": 4, "min": 2, "max": 32, "step": .1}),

                **QUILTING_SHARED_INPUT_TYPES
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "compute"
    CATEGORY = "Bmad/CV/Misc"

    def compute(self, src, block_size, scale, overlap, tolerance, parallelization_lvl, seed):
        src = src["samples"]
        H, W = src.shape[2:4]
        outH, outW = int(scale * H), int(scale * W)
        print(f"shape = {src.shape}")
        print(f"sizes {(H, W, outH, outW)}")
        if overlap > 0:
            overlap = int(block_size * overlap)
        else:
            overlap = int(block_size / 6.0)

        # uses normalized values, not 0 to 255

        rng: numpy.random.Generator = np.random.default_rng(seed=seed)

        finish_event, t, shm_name, shm_jobs = \
            setup_pbar(block_size, overlap, outH, outW, parallelization_lvl, src.shape[0])

        if src.shape[0] > 1:  # if multiple
            latent_batch = self.batch_using_jobs(src, block_size, overlap, outH, outW, tolerance,
                                                 parallelization_lvl, rng, shm_name)
            terminate_generation(finish_event, shm_jobs, t)
            return ({"samples": latent_batch},)

        # if single
        src = src[0].cpu().numpy().squeeze()
        src = np.moveaxis(src, 0, -1)

        if parallelization_lvl == 0:
            from .quilting.generate import generateTextureMap
            texture = generateTextureMap(src, block_size, overlap, outH, outW, tolerance, rng, shm_name, 0)
        else:
            from .parallel_quilting import generateTextureMap_p
            texture = generateTextureMap_p(src, block_size, overlap, outH, outW, tolerance, parallelization_lvl, rng,
                                           shm_name, 0)

        terminate_generation(finish_event, shm_jobs, t)
        texture = np.moveaxis(texture, -1, 0)
        texture = torch.from_numpy(texture).unsqueeze(0)
        return ({"samples": texture},)

    @staticmethod
    def batch_using_jobs(src, block_size, overlap, outH, outW, tolerance, parallelization_lvl, rng, jobs_shm_name):
        from joblib import Parallel, delayed

        def unwrap_and_quilt(latent, job_id):
            latent = latent.cpu().numpy().squeeze()
            latent = np.moveaxis(latent, 0, -1)
            if parallelization_lvl == 0:
                result = quilt_single_src_no_parallelization(latent, block_size, overlap, outH, outW, tolerance, rng,
                                                             jobs_shm_name, job_id)
            else:
                result = quilt_single_with_parallelization(latent, block_size, overlap, outH, outW, tolerance, 1, rng,
                                                           jobs_shm_name, job_id)
            result = np.moveaxis(result, -1, 0)
            return torch.from_numpy(result)

        results = Parallel(n_jobs=-1, backend="loky", timeout=None)(
            delayed(unwrap_and_quilt)(src[i], i) for i in range(src.shape[0]))

        return torch.stack(results)
