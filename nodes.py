from .make_seamless import (make_seamless_horizontally, make_seamless_vertically,
                            make_seamless_both, get_numb_of_blocks_to_fill_stripe)
from .quilting import generate_texture, generate_texture_parallel
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Event
from .types import UiCoordData
from threading import Thread
from comfy import utils
import numpy as np
import numpy.random
import torch
import cv2

# TODO add nodes where user defines output's height and width instead of scale

NODES_CATEGORY = "Bmad/CV/Quilting"


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
    "tolerance": ("FLOAT", {"default": .1, "min": 0, "max": 2, "step": .01}),

    # 0 -> no parallelization; uses the reference implementation
    # 1 -> 4 jobs used ( divides generation into 4 sections )
    # 2 and above ->  the number of jobs per section
    # parallelization lvl also affects output; even if seed is fixed, changing this will change the output
    "parallelization_lvl": ("INT", {"default": 1, "min": 0, "max": 6, "step": 1}),

    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),

    "version": ("INT", {"default": 1, "min": 0, "max": 3}),
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


def setup_pbar_quilting(block_size, overlap, out_height, out_width, par_lvl, batch_len):
    n_rows = int(np.ceil((out_height - block_size) * 1.0 / (block_size - overlap)))
    n_columns = int(np.ceil((out_width - block_size) * 1.0 / (block_size - overlap)))
    total_steps: int = ((n_rows + 1) * (n_columns + 1) - 1) * batch_len  # ignores first corner/center
    # might be less than the generated when using parallel solution, but not by far, so will leave it like this for now

    return setup_pbar(total_steps, par_lvl, batch_len)


def setup_pbar_seamless(ori, block_size, overlap, height, width, batch_len):
    match ori:
        case "H":
            total_steps = get_numb_of_blocks_to_fill_stripe(block_size, overlap, width)
        case "V":
            total_steps = get_numb_of_blocks_to_fill_stripe(block_size, overlap, height)
        case _:
            total_steps = (
                2 +
                get_numb_of_blocks_to_fill_stripe(block_size, overlap, height) +
                get_numb_of_blocks_to_fill_stripe(block_size, overlap, width)
            )
    return setup_pbar(total_steps, 0, batch_len)


def setup_pbar(total_steps, par_lvl, batch_len):
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
    def INPUT_TYPES(cls):
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
    CATEGORY = NODES_CATEGORY

    def compute(self, src, block_size, scale, overlap, tolerance, parallelization_lvl, seed, version):
        h, w = src.shape[1:3]
        out_h, out_w = int(scale * h), int(scale * w)
        overlap = int(block_size * overlap) if overlap > 0 else int(
            block_size * QUILTING_SHARED_INPUT_TYPES["overlap"][1]["default"])

        # note: the input src should have normalized values, not 0 to 255

        rng: numpy.random.Generator = np.random.default_rng(seed=seed)

        finish_event, t, shm_name, shm_jobs = \
            setup_pbar_quilting(block_size, overlap, out_h, out_w, parallelization_lvl, src.shape[0])

        if src.shape[0] > 1:  # if image batch
            texture_batch = self.batch_using_jobs(src, block_size, overlap, out_h, out_w, tolerance, version,
                                                  parallelization_lvl, rng, shm_name)
            terminate_generation(finish_event, shm_jobs, t)
            return (texture_batch,)

        # ____ if single image
        src = src.cpu().numpy().squeeze()
        if version == 2:
            src = cv2.cvtColor(src, cv2.COLOR_RGB2Lab)

        if parallelization_lvl == 0:
            texture = generate_texture(
                src, block_size, overlap, out_h, out_w, tolerance, version, rng,
                UiCoordData(shm_name, 0))
        else:
            texture = generate_texture_parallel(
                src, block_size, overlap, out_h, out_w, tolerance, version, parallelization_lvl, rng,
                UiCoordData(shm_name, 0))

        if version == 2:
            texture = cv2.cvtColor(texture, cv2.COLOR_LAB2RGB)
        texture = torch.from_numpy(texture).unsqueeze(0)

        terminate_generation(finish_event, shm_jobs, t)
        return (texture,)

    @staticmethod
    def batch_using_jobs(src, block_size, overlap, outH, outW, tolerance, version, parallelization_lvl, rng, jobs_shm_name):
        from joblib import Parallel, delayed

        def unwrap_and_quilt(img_as_tensor, job_id):
            image = img_as_tensor.cpu().numpy()
            if version == 2:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

            if parallelization_lvl == 0:
                result = generate_texture(image, block_size, overlap, outH, outW, tolerance, version, rng,
                                          UiCoordData(jobs_shm_name, job_id))
            else:
                result = generate_texture_parallel(image, block_size, overlap, outH, outW, tolerance, version, 1, rng,
                                                   UiCoordData(jobs_shm_name, job_id))

            if version == 2:
                result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
            return torch.from_numpy(result)

        results = Parallel(n_jobs=-1, backend="loky", timeout=None)(
            delayed(unwrap_and_quilt)(src[i], i) for i in range(src.shape[0]))

        return torch.stack(results)


class LatentQuilting:
    @classmethod
    def INPUT_TYPES(cls):
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
    CATEGORY = NODES_CATEGORY

    def compute(self, src, block_size, scale, overlap, tolerance, parallelization_lvl, seed, version):
        src = src["samples"]
        h, w = src.shape[2:4]
        out_h, out_w = int(scale * h), int(scale * w)
        overlap = int(block_size * overlap) if overlap > 0 else int(
            block_size * QUILTING_SHARED_INPUT_TYPES["overlap"][1]["default"])

        rng: numpy.random.Generator = np.random.default_rng(seed=seed)

        finish_event, t, shm_name, shm_jobs = \
            setup_pbar_quilting(block_size, overlap, out_h, out_w, parallelization_lvl, src.shape[0])

        if src.shape[0] > 1:  # if multiple
            latent_batch = self.batch_using_jobs(src, block_size, overlap, out_h, out_w, tolerance, version,
                                                 parallelization_lvl, rng, shm_name)
            terminate_generation(finish_event, shm_jobs, t)
            return ({"samples": latent_batch},)

        # ____ if single
        src = src[0].cpu().numpy().squeeze()
        src = np.moveaxis(src, 0, -1)

        if parallelization_lvl == 0:
            texture = generate_texture(
                src, block_size, overlap, out_h, out_w, tolerance, version, rng,
                UiCoordData(shm_name, 0))
        else:
            texture = generate_texture_parallel(
                src, block_size, overlap, out_h, out_w, tolerance, version, parallelization_lvl, rng,
                UiCoordData(shm_name, 0))

        terminate_generation(finish_event, shm_jobs, t)
        texture = np.moveaxis(texture, -1, 0)
        texture = torch.from_numpy(texture).unsqueeze(0)
        return ({"samples": texture},)

    @staticmethod
    def batch_using_jobs(src, block_size, overlap, outH, outW, tolerance, version, parallelization_lvl, rng, jobs_shm_name):
        from joblib import Parallel, delayed

        def unwrap_and_quilt(latent, job_id):
            latent = latent.cpu().numpy().squeeze()
            latent = np.moveaxis(latent, 0, -1)
            if parallelization_lvl == 0:
                result = generate_texture(
                    latent, block_size, overlap, outH, outW, tolerance, version, rng,
                    UiCoordData(jobs_shm_name, job_id))
            else:
                result = generate_texture_parallel(
                    latent, block_size, overlap, outH, outW, tolerance, version, 1, rng,
                    UiCoordData(jobs_shm_name, job_id))
            result = np.moveaxis(result, -1, 0)
            return torch.from_numpy(result)

        results = Parallel(n_jobs=-1, backend="loky", timeout=None)(
            delayed(unwrap_and_quilt)(src[i], i) for i in range(src.shape[0]))

        return torch.stack(results)


class ImageMakeSeamlessMB:
    """
    Transition stripe is built using overlapping square patches.
    """
    SEAMLESS_DIR = {
        "H": make_seamless_horizontally,
        "V": make_seamless_vertically,
        "H & V": make_seamless_both
    }

    @classmethod
    def INPUT_TYPES(cls):
        inputs = QUILTING_SHARED_INPUT_TYPES.copy()
        inputs.pop("parallelization_lvl")
        return {
            "required": {
                "src": ("IMAGE",),
                "ori": (list(cls.SEAMLESS_DIR.keys()), {"default": "H"}),
                **inputs,
            },
            "optional": {
                "lookup": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compute"
    CATEGORY = NODES_CATEGORY

    def compute(self, src, ori, block_size, overlap, tolerance, seed, version, lookup=None):
        # note that src = lookup is the current algorithm policy when lookup is not provided.
        # this policy could change in the future, so do not apply it here too despite being idempotent.
        h, w = src.shape[1:3]
        overlap = int(block_size * overlap) if overlap > 0 else int(
            block_size * QUILTING_SHARED_INPUT_TYPES["overlap"][1]["default"])
        rng: numpy.random.Generator = np.random.default_rng(seed=seed)

        finish_event, t, shm_name, shm_jobs = \
            setup_pbar_seamless(ori, block_size, overlap, h, w, 0)
        uicd = UiCoordData(shm_name, 0)

        if src.shape[0] > 1:  # if image batch
            return (None,)   # TODO

        # ____ if single image
        src = src.cpu().numpy().squeeze()
        lookup = lookup.cpu().numpy().squeeze() if lookup is not None else None
        if version == 2:
            src = cv2.cvtColor(src, cv2.COLOR_RGB2Lab)
            if lookup is not None:
                lookup = cv2.cvtColor(lookup, cv2.COLOR_RGB2Lab) if lookup is not None else None

        seamless_func = self.SEAMLESS_DIR[ori]
        output = seamless_func(src, block_size, overlap, tolerance, rng, version, lookup, uicd)
        if version == 2:
            output = cv2.cvtColor(output, cv2.COLOR_LAB2RGB)
        output = torch.from_numpy(output).unsqueeze(0)
        terminate_generation(finish_event, shm_jobs, t)
        return (output, )
