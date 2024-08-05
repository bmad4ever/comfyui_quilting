from .quilting import generate_texture, generate_texture_parallel
from .guess_block_size import guess_nice_block_size
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Event
from dataclasses import dataclass
from .types import UiCoordData
from threading import Thread
from comfy import utils
import numpy.random
import numpy as np
import torch
import cv2

# TODO add nodes where user defines output's height and width instead of scale

NODES_CATEGORY = "Bmad/CV/Quilting"
SEAMLESS_DIRS = ["H", "V", "H & V"]  # options for seamless nodes


def get_quilting_shared_input_types():
    return {
        # block size is given in pixels.
        # if inferior to 3, guess_nice_block_size is used instead.
        # if negative, guess_nice_block_size only does freq. analysis (should be considerably faster)
        "block_size": ("INT", {"default": -1, "min": -1, "max": 512, "step": 1}),

        # the percentage of pixels that overlap between each block_sized block
        "overlap": ("FLOAT", {"default": 1 / 5.0, "min": .1, "max": .9, "step": .01}),

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


@dataclass
class QuiltingFuncWrapper:
    """Wraps node functionality for easy re-use when using jobs."""
    block_sizes: list[int]
    overlap: float
    out_h: int
    out_w: int
    tolerance: float
    parallelization_lvl: int
    rng: numpy.random.Generator
    version: int
    jobs_shm_name: str


# region AUX FUNCTIONS
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


def terminate_task(finished_event, jobs_shared_memory, pbt: Thread):
    """
    1. triggers finished_event so that UI thread can stop looping
    2. check if the process stopped due to an interruption
    3. closes/unlinks shared memory
    4. if process stopped due to interruption executes throw_exception_if_processing_interrupted()
    """
    coord_jobs_array = np.ndarray((1,), dtype=np.dtype('uint32'), buffer=jobs_shared_memory.buf)
    interrupted = coord_jobs_array[0]
    finished_event.set()
    pbt.join()
    jobs_shared_memory.close()
    jobs_shared_memory.unlink()
    if interrupted:
        from comfy.model_management import throw_exception_if_processing_interrupted
        throw_exception_if_processing_interrupted()


def setup_pbar_quilting(block_sizes: list[int], overlap_percentage: float, out_height, out_width, par_lvl):
    """
    @param block_sizes: block size for each item in the batch. len(block_sizes) = batch length
    """
    total_steps: int = 0
    batch_len: int = len(block_sizes)

    for block_size in block_sizes:
        overlap = overlap_percentage_to_pixels(block_size, overlap_percentage)
        n_rows = int(np.ceil((out_height - block_size) * 1.0 / (block_size - overlap)))
        n_columns = int(np.ceil((out_width - block_size) * 1.0 / (block_size - overlap)))
        total_steps += ((n_rows + 1) * (n_columns + 1) - 1)  # ignores first corner/center
    # might be less than the generated when using parallel solution, but not by far, so will leave it like this for now

    return setup_pbar(total_steps, par_lvl, batch_len)


def setup_pbar_seamless(ori, block_sizes: list[int], overlap_percentage: float, height, width, batch_len):
    """
    here batch_len may differ from block_sizes due to the number of lookup textures
    """
    from .make_seamless import get_numb_of_blocks_to_fill_stripe
    total_steps: int = 0
    for i in range(batch_len):
        block_size = block_sizes[min(i, len(block_sizes) - 1)]
        overlap = overlap_percentage_to_pixels(block_size, overlap_percentage)
        match ori:
            case "H":
                total_steps += get_numb_of_blocks_to_fill_stripe(block_size, overlap, width)
            case "V":
                total_steps += get_numb_of_blocks_to_fill_stripe(block_size, overlap, height)
            case _:
                total_steps += (
                        2 +
                        get_numb_of_blocks_to_fill_stripe(block_size, overlap, height) +
                        get_numb_of_blocks_to_fill_stripe(block_size, overlap, width)
                )
    return setup_pbar(total_steps, 0, batch_len)


def setup_pbar_seamless_v2(ori, batch_len):
    # 3 increments per big block + 2 for the "H & V" H Seam patch
    match ori:
        case "H & V":
            total_steps = 2 + 3 * 2
        case _:
            total_steps = 3
    total_steps = total_steps * batch_len
    return setup_pbar(total_steps, 0, batch_len)


def setup_pbar(total_steps, par_lvl, batch_len):
    """
    @param total_steps: aggregate from all the jobs/batch items; not for a single job or batch item
    """
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


def unwrap_to_grey(image, is_latent: bool = False) -> np.ndarray:
    squeeze = len(image.shape) > 3
    image = image.cpu().numpy()
    image = image.squeeze() if squeeze else image
    image = np.moveaxis(image, 0, -1) if is_latent else image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def unwrap_and_quilt(wrapped_func, image, job_id, is_latent: bool = False):
    """quilting job when using batches"""
    squeeze = len(image.shape) > 3
    image = image.cpu().numpy()
    image = image.squeeze() if squeeze else image
    image = np.moveaxis(image, 0, -1) if is_latent else image
    result = wrapped_func(image, job_id)
    result = np.moveaxis(result, -1, 0) if is_latent else result
    result = torch.from_numpy(result)
    result = result.unsqueeze(0) if squeeze else result
    return result


def unwrap_and_quilt_seamless(wrapped_func, image, lookup, job_id):
    """seamless quilting job when using batches"""
    image = image.cpu().numpy()
    squeeze = len(image.shape) > 3
    image = image.squeeze() if squeeze else image
    if lookup is not None:
        lookup = lookup.cpu().numpy()
        lookup = lookup.squeeze() if len(lookup.shape) > 3 else lookup
    result = wrapped_func(image, lookup, job_id)
    result = torch.from_numpy(result)
    result = result.unsqueeze(0) if squeeze else result
    return result


def overlap_percentage_to_pixels(block_size: int, overlap: float):
    return int(block_size * overlap) if overlap > 0 else int(
        block_size * get_quilting_shared_input_types()["overlap"][1]["default"])


def get_block_sizes(src, block_size_input: int, block_size_upper_bound: int|None = None):
    if block_size_input >= 3:
        return [block_size_input] * src.shape[0]

    print(f"block size set to {block_size_input} less than 3!\nguessing nice block size...")
    only_do_freq_analysis = block_size_input < 0
    sizes = [guess_nice_block_size(unwrap_to_grey(src[i]), only_do_freq_analysis, block_size_upper_bound)
             for i in range(src.shape[0])]
    print(f"guessed block sizes: {sizes}")

    return sizes


def block_size_upper_bound_for_seamless(ori, tex_h, tex_w, overlap_percentage):
    """
    if using guess_block_size in seamless nodes, H seam
    """
    match ori:
        case "H & V":
            # mind H seam w/ the following related assert -> image.shape[1] >= block_size + overlap * 2
            # -> H >= S + S * OP * 2 <-> H / (1 + 2 * OP ) >= S
            return min(
                # similar value to the one computed in guess nice block
                round(min(tex_w, tex_h) / 1.2),
                # space required to patch h seam  plus some extra margin
                round(tex_h/(1+2*overlap_percentage)/1.1)
            )
        case _______:
            return None  # texture default dims are fine as bounds

# endregion AUX FUNCTIONS & CLASSES


# region NODES

class ImageQuilting:
    class ImageQuiltingFuncWrapper(QuiltingFuncWrapper):
        """Wraps node functionality for easy re-use when using jobs."""

        def __call__(self, image, job_id):
            if self.version == 2:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

            block_size = self.block_sizes[job_id]
            overlap = overlap_percentage_to_pixels(block_size, self.overlap)

            if self.parallelization_lvl == 0:
                result = generate_texture(
                    image, block_size, overlap, self.out_h, self.out_w, self.tolerance,
                    self.version, self.rng, UiCoordData(self.jobs_shm_name, job_id))
            else:
                result = generate_texture_parallel(
                    image, block_size, overlap, self.out_h, self.out_w, self.tolerance,
                    self.version, self.parallelization_lvl, self.rng, UiCoordData(self.jobs_shm_name, job_id))

            if self.version == 2:
                result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

            return result

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src": ("IMAGE",),
                # can be a single image or a batch of images
                # using an image batch limits max parallelization lvl to 1 ( values above are ignored ).

                # self explanatory
                "scale": ("FLOAT", {"default": 4, "min": 2, "max": 10, "step": .1}),

                **get_quilting_shared_input_types()
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compute"
    CATEGORY = NODES_CATEGORY

    def compute(self, src, block_size, scale, overlap, tolerance, parallelization_lvl, seed, version):
        h, w = src.shape[1:3]
        out_h, out_w = int(scale * h), int(scale * w)
        block_sizes = get_block_sizes(src, block_size)
        rng: numpy.random.Generator = np.random.default_rng(seed=seed)

        # note: the input src should have normalized values, not 0 to 255

        finish_event, t, shm_name, shm_jobs = \
            setup_pbar_quilting(block_sizes, overlap, out_h, out_w, parallelization_lvl)

        try:
            func = ImageQuilting.ImageQuiltingFuncWrapper(
                block_sizes, overlap, out_h, out_w, tolerance, parallelization_lvl, rng, version, shm_name)

            is_batch = src.shape[0] > 1
            output = self.batch_using_jobs(func, src) if is_batch else unwrap_and_quilt(func, src, 0)
        except Exception as ex:
            raise ex
        finally:
            terminate_task(finish_event, shm_jobs, t)
        return (output,)

    @staticmethod
    def batch_using_jobs(wrapped_func, src):
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=-1, backend="loky", timeout=None)(
            delayed(unwrap_and_quilt)(wrapped_func, src[i], i) for i in range(src.shape[0]))
        return torch.stack(results)


class LatentQuilting:
    class LatentQuiltingFuncWrapper(QuiltingFuncWrapper):
        """Wraps node functionality for easy re-use when using jobs."""

        def __call__(self, latent_image, job_id):
            block_size = self.block_sizes[job_id]
            overlap = overlap_percentage_to_pixels(block_size, self.overlap)
            if self.parallelization_lvl == 0:
                return generate_texture(
                    latent_image, block_size, overlap, self.out_h, self.out_w, self.tolerance,
                    self.version, self.rng, UiCoordData(self.jobs_shm_name, job_id))
            else:
                return generate_texture_parallel(
                    latent_image, block_size, overlap, self.out_h, self.out_w, self.tolerance,
                    self.version, self.parallelization_lvl, self.rng, UiCoordData(self.jobs_shm_name, job_id))

    @classmethod
    def INPUT_TYPES(cls):
        shared_inputs = get_quilting_shared_input_types()
        # don't allow for auto block size
        shared_inputs["block_size"][1]["default"] = 18
        shared_inputs["block_size"][1]["min"] = 3
        # note: could try running over individual channels; not sure if worth it thought.

        return {
            "required": {
                "src": ("LATENT",),

                # self explanatory
                "scale": ("FLOAT", {"default": 4, "min": 2, "max": 32, "step": .1}),

                **shared_inputs
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "compute"
    CATEGORY = NODES_CATEGORY

    def compute(self, src, block_size, scale, overlap, tolerance, parallelization_lvl, seed, version):
        src = src["samples"]
        h, w = src.shape[2:4]
        out_h, out_w = int(scale * h), int(scale * w)
        block_sizes = [block_size] * src.shape[0]
        rng: numpy.random.Generator = np.random.default_rng(seed=seed)

        finish_event, t, shm_name, shm_jobs = \
            setup_pbar_quilting(block_sizes, overlap, out_h, out_w, parallelization_lvl)

        try:
            func = LatentQuilting.LatentQuiltingFuncWrapper(
                block_sizes, overlap, out_h, out_w, tolerance, parallelization_lvl, rng, version, shm_name)

            is_batch = src.shape[0] > 1
            output = self.batch_using_jobs(func, src) if is_batch else\
                unwrap_and_quilt(func, src, 0, is_latent=True)
        except Exception as ex:
            raise ex
        finally:
            terminate_task(finish_event, shm_jobs, t)
        return ({"samples": output},)

    @staticmethod
    def batch_using_jobs(wrapped_func, src):
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=-1, backend="loky", timeout=None)(
            delayed(unwrap_and_quilt)(wrapped_func, src[i], i, True) for i in range(src.shape[0]))
        return torch.stack(results)


class ImageMakeSeamlessMB:
    """Transition stripe is built using overlapping square patches."""

    @dataclass
    class SeamlessFuncWrapper:
        """Wraps node functionality for easy re-use when using jobs."""

        ori: str
        block_sizes: list[int]
        overlap: int
        tolerance: float
        rng: numpy.random.Generator
        version: int
        jobs_shm_name: str

        def __call__(self, image, lookup, job_id):
            from .make_seamless import make_seamless_horizontally, make_seamless_vertically, make_seamless_both

            block_size = self.block_sizes[min(job_id, len(self.block_sizes) - 1)]
            overlap = overlap_percentage_to_pixels(block_size, self.overlap)

            if self.version == 2:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
                if lookup is not None:
                    lookup = cv2.cvtColor(lookup, cv2.COLOR_RGB2Lab) if lookup is not None else None

            match self.ori:
                case "H":
                    func = make_seamless_horizontally
                case "V":
                    func = make_seamless_vertically
                case ___:
                    func = make_seamless_both

            result = func(image, block_size, overlap, self.tolerance,
                          self.rng, self.version, lookup, UiCoordData(self.jobs_shm_name, job_id))

            if self.version == 2:
                result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

            return result

    @classmethod
    def INPUT_TYPES(cls):
        inputs = get_quilting_shared_input_types()
        inputs.pop("parallelization_lvl")
        inputs["version"][1]["min"] = 1
        inputs["overlap"][1]["max"] = .5
        return {
            "required": {
                "src": ("IMAGE",),
                "ori": (SEAMLESS_DIRS, {"default": SEAMLESS_DIRS[0]}),
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
        blk_size_upper_bound = block_size_upper_bound_for_seamless(ori, h, w, overlap)
        block_sizes = get_block_sizes(src, block_size, blk_size_upper_bound)
        rng: numpy.random.Generator = np.random.default_rng(seed=seed)

        lookup_batch_size = lookup.shape[0] if lookup is not None else 0
        finish_event, t, shm_name, shm_jobs = setup_pbar_seamless(
            ori, block_sizes, overlap, h, w, max(src.shape[0], lookup_batch_size))

        try:
            func = ImageMakeSeamlessMB.SeamlessFuncWrapper(
                ori, block_sizes, overlap, tolerance, rng, version, shm_name)

            is_batch = src.shape[0] > 1 or lookup_batch_size > 1
            output = self.batch_using_jobs(func, src, lookup) if is_batch else\
                unwrap_and_quilt_seamless(func, src, lookup, 0)
        except Exception as ex:
            raise ex
        finally:
            terminate_task(finish_event, shm_jobs, t)
        return (output,)

    @staticmethod
    def batch_using_jobs(wrapped_func, src, lookup):
        from joblib import Parallel, delayed
        # process in the same fashion as lists
        number_of_processes = src.shape[0] if lookup is None else max(src.shape[0], lookup.shape[0])
        results = Parallel(n_jobs=-1, backend="loky", timeout=None)(
            delayed(unwrap_and_quilt_seamless)(
                wrapped_func,
                src[min(i, src.shape[0] - 1)],
                lookup[min(i, lookup.shape[0] - 1)]
                if lookup is not None else None,
                i) for i in range(number_of_processes))
        return torch.stack(results)


class ImageMakeSeamlessSB:
    """Transition stripe is built via a single rectangular block."""

    @dataclass
    class SeamlessFuncWrapper:
        """Wraps node functionality for easy re-use when using jobs."""

        ori: str
        block_sizes: list[int]
        overlap: int
        rng: numpy.random.Generator
        version: int
        jobs_shm_name: str

        def __call__(self, image, lookup, job_id):
            from .make_seamless2 import seamless_horizontal, seamless_vertical, seamless_both

            block_size = self.block_sizes[min(job_id, len(self.block_sizes) - 1)]
            overlap = overlap_percentage_to_pixels(block_size, self.overlap)

            if self.version == 2:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
                if lookup is not None:
                    lookup = cv2.cvtColor(lookup, cv2.COLOR_RGB2Lab) if lookup is not None else None

            match self.ori:
                case "H":
                    func = seamless_horizontal
                case "V":
                    func = seamless_vertical
                case ___:
                    func = seamless_both

            result = func(image, block_size, overlap,
                          self.version, lookup, self.rng, UiCoordData(self.jobs_shm_name, job_id))

            if self.version == 2:
                result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

            return result

    @classmethod
    def INPUT_TYPES(cls):
        inputs = get_quilting_shared_input_types()
        inputs.pop("parallelization_lvl")
        inputs.pop("tolerance")
        inputs["version"][1]["min"] = 1
        inputs["overlap"][1]["max"] = .5
        return {
            "required": {
                "src": ("IMAGE",),
                "ori": (SEAMLESS_DIRS, {"default": SEAMLESS_DIRS[0]}),
                **inputs,
            },
            "optional": {
                "lookup": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compute"
    CATEGORY = NODES_CATEGORY

    def compute(self, src, ori, block_size, overlap, seed, version, lookup=None):
        # note that src = lookup is the current algorithm policy when lookup is not provided.
        # this policy could change in the future, so do not apply it here too despite being idempotent.
        h, w = src.shape[1:3]
        blk_size_upper_bound = block_size_upper_bound_for_seamless(ori, h, w, overlap)
        block_sizes = get_block_sizes(src, block_size, blk_size_upper_bound)
        rng: numpy.random.Generator = np.random.default_rng(seed=seed)

        lookup_batch_size = lookup.shape[0] if lookup is not None else 0
        finish_event, t, shm_name, shm_jobs = setup_pbar_seamless_v2(ori, max(src.shape[0], lookup_batch_size))

        try:
            func = ImageMakeSeamlessSB.SeamlessFuncWrapper(
                ori, block_sizes, overlap, rng, version, shm_name)

            is_batch = src.shape[0] > 1 or lookup_batch_size > 1
            output = self.batch_using_jobs(func, src, lookup) if is_batch else\
                unwrap_and_quilt_seamless(func, src, lookup, 0)
        except Exception as ex:
            raise ex
        finally:
            terminate_task(finish_event, shm_jobs, t)
        return (output,)

    @staticmethod
    def batch_using_jobs(wrapped_func, src, lookup):
        from joblib import Parallel, delayed
        # process in the same fashion as lists
        number_of_processes = src.shape[0] if lookup is None else max(src.shape[0], lookup.shape[0])
        results = Parallel(n_jobs=-1, backend="loky", timeout=None)(
            delayed(unwrap_and_quilt_seamless)(
                wrapped_func,
                src[min(i, src.shape[0] - 1)],
                lookup[min(i, lookup.shape[0] - 1)]
                if lookup is not None else None,
                i) for i in range(number_of_processes))
        return torch.stack(results)


class GuessNiceBlockSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src": ("IMAGE",),
                "simple_and_fast": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "compute"
    CATEGORY = NODES_CATEGORY

    def compute(self, src, simple_and_fast):
        return (guess_nice_block_size(unwrap_to_grey(src[0]), freq_analysis_only=simple_and_fast),)

# endregion NODES
