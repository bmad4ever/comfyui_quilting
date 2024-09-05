from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Event
from threading import Thread
from comfy import utils
import numpy.random
import numpy as np
import cv2

from bmquilting.misc.validation_utils import validate_array_shape
from bmquilting.guess_block_size import guess_nice_block_size
from bmquilting.types import Orientation, UiCoordData
from bmquilting.comfy_utils.utilities import (
    unwrap_latent_and_quilt, unwrap_latent_and_quilt_seamless, overlap_percentage_to_pixels)
from bmquilting.comfy_utils.wrapped_functions import (
    QuiltingFuncWrapper, batch_using_jobs,
    SeamlessMPFuncWrapper, SeamlessSPFuncWrapper, batch_seamless_using_jobs)

# TODO add nodes where user defines output's height and width instead of scale

NODES_CATEGORY = "Bmad/CV/Quilting"
SEAMLESS_DIRS = [orientation.value for orientation in Orientation]  # options for seamless nodes
v0_min_tolerance = .001  # if set to zero when using version zero, use this value instead


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

        "version": ("INT", {"default": 3, "min": 0, "max": 3}),

        "blend_into_patch": ("BOOLEAN", {"default": False})
    }


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
    1. clear methods cache
    2. triggers finished_event so that UI thread can stop looping
    3. check if the process stopped due to an interruption
    4. closes/unlinks shared memory
    5. if process stopped due to interruption executes throw_exception_if_processing_interrupted()
    """
    from bmquilting.synthesis_subroutines import patch_blending_vignette
    patch_blending_vignette.cache_clear()

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
    from bmquilting.make_seamless import get_numb_of_blocks_to_fill_stripe
    total_steps: int = 0
    for i in range(batch_len):
        block_size = block_sizes[min(i, len(block_sizes) - 1)]
        overlap = overlap_percentage_to_pixels(block_size, overlap_percentage)
        match ori:
            case Orientation.H:
                total_steps += get_numb_of_blocks_to_fill_stripe(block_size, overlap, width)
            case Orientation.V:
                total_steps += get_numb_of_blocks_to_fill_stripe(block_size, overlap, height)
            case _:
                total_steps += (
                        2 +
                        get_numb_of_blocks_to_fill_stripe(block_size, overlap, height) +
                        get_numb_of_blocks_to_fill_stripe(block_size, overlap, width)
                )
    return setup_pbar(total_steps, 0, batch_len)


def setup_pbar_seamless_v2(ori: Orientation, batch_len: int):
    # 3 increments per big block + 2 for the "H & V" H Seam patch
    match ori:
        case Orientation.H_AND_V:
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

    shm_size, n_jobs = UiCoordData.get_required_shm_size_and_number_of_jobs(par_lvl, batch_len)
    shm_jobs = SharedMemory(create=True, size=shm_size)

    t = Thread(target=waiting_loop, args=(finished_event, pbar, total_steps, shm_jobs.name, n_jobs))
    t.start()

    return finished_event, t, shm_jobs.name, shm_jobs


def unwrap_to_grey(image, is_latent: bool = False) -> np.ndarray:
    squeeze = len(image.shape) > 3
    image = image.cpu().numpy()
    image = image.squeeze() if squeeze else image
    image = np.moveaxis(image, 0, -1) if is_latent else image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def get_block_sizes(src, block_size_input: int, block_size_upper_bound: int | None = None):
    match block_size_input:
        case _ if block_size_input in [-1, 0]:
            print(f"block size set to {block_size_input}!\nguessing nice block size...")
            only_do_freq_analysis = block_size_input < 0
            sizes = [guess_nice_block_size(unwrap_to_grey(src[i]), only_do_freq_analysis, block_size_upper_bound)
                     for i in range(src.shape[0])]
        case 1:
            print(f"src shape = {src.shape}")
            sizes = [round(min(src.shape[1:3]) * 1 / 3)] * src.shape[0]  # a "medium" block size, w/ respect to src
        case 2:
            sizes = [round(min(src.shape[1:3]) * 3 / 4)] * src.shape[0]  # a "big" block size w/ respect to src
        case __:
            sizes = [block_size_input] * src.shape[0]
    print(f"block sizes: {sizes}")
    return sizes


def block_size_upper_bound_for_seamless(ori, tex_h, overlap_percentage):
    """
    if using guess_block_size in seamless nodes, mind H seam.
    note that if the default computed upper bound is lower than the value here obtained, it will be used instead.
    """
    match ori:
        case "H & V":
            # mind H seam w/ the following related assert -> image.shape[1] >= block_size + overlap * 2
            # -> H >= S + S * OP * 2 <-> H / (1 + 2 * OP ) >= S
            return round(tex_h / (1 + 2 * overlap_percentage) / 1.1)
        case _______:
            return None  # texture default dims are fine as bounds


def validate_seamless_args(orientation, source, lookup, block_size, overlap):
    if overlap * 2 > block_size:
        raise ValueError(f"Overlap ({overlap}) needs to be 50% or less of the block size ({block_size}).")

    if orientation == "H & V":
        validate_array_shape(source, min_height=block_size, min_width=block_size + overlap * 2,
                             help_msg="Change the block size or the overlap.")
    else:
        validate_array_shape(source, min_height=block_size, min_width=block_size,
                             help_msg="Change the block size.")

    if lookup is not None:
        validate_array_shape(lookup, min_height=block_size, min_width=block_size,
                             help_msg="Use a bigger lookup or change the block size.")
        if lookup.dtype != source.dtype:
            raise TypeError("lookup_texture dtype does not match image dtype")


# endregion AUX FUNCTIONS & CLASSES


# region NODES

class ImageQuilting:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src": ("IMAGE",),
                # can be a single image or a batch of images
                # using an image batch limits max parallelization lvl to 1 ( values above are ignored ).

                "scale": ("FLOAT", {"default": 4, "min": 2, "max": 32, "step": .1}),  # self-explanatory

                **get_quilting_shared_input_types()
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compute"
    CATEGORY = NODES_CATEGORY

    def compute(self, src, block_size, scale, overlap, tolerance, parallelization_lvl, seed, version, blend_into_patch):
        h, w = src.shape[1:3]
        out_h, out_w = int(scale * h), int(scale * w)
        block_sizes = get_block_sizes(src, block_size)
        rng: numpy.random.Generator = np.random.default_rng(seed=seed)
        if version == 0 and tolerance < 0.01:
            tolerance = v0_min_tolerance

        # note: the input src should have normalized values, not 0 to 255

        finish_event, t, shm_name, shm_jobs = \
            setup_pbar_quilting(block_sizes, overlap, out_h, out_w, parallelization_lvl)

        try:
            func = QuiltingFuncWrapper(False,
                block_sizes, overlap, out_h, out_w, tolerance,
                parallelization_lvl, rng, blend_into_patch, version, shm_name)

            is_batch = src.shape[0] > 1
            output = batch_using_jobs(func, src, is_latent=False) if is_batch\
                else unwrap_latent_and_quilt(func, src, is_latent=False)
        finally:
            terminate_task(finish_event, shm_jobs, t)
        return (output,)


class LatentQuilting:
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

    def compute(self, src, block_size, scale, overlap, tolerance, parallelization_lvl, seed, version, blend_into_patch):
        src = src["samples"]
        h, w = src.shape[2:4]
        out_h, out_w = int(scale * h), int(scale * w)
        block_sizes = [block_size] * src.shape[0]
        rng: numpy.random.Generator = np.random.default_rng(seed=seed)
        if version == 0 and tolerance < 0.01:
            tolerance = v0_min_tolerance

        finish_event, t, shm_name, shm_jobs = \
            setup_pbar_quilting(block_sizes, overlap, out_h, out_w, parallelization_lvl)

        try:
            func = QuiltingFuncWrapper(True,
                block_sizes, overlap, out_h, out_w, tolerance, parallelization_lvl,
                rng, blend_into_patch, version, shm_name)

            is_batch = src.shape[0] > 1
            output = batch_using_jobs(func, src, is_latent=True) if is_batch else \
                unwrap_latent_and_quilt(func, src, is_latent=True)
        finally:
            terminate_task(finish_event, shm_jobs, t)
        return ({"samples": output},)


class ImageMakeSeamlessMB:
    """Transition stripe is built using overlapping square patches."""

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

    def compute(self, src, ori, block_size, overlap, tolerance, seed, version, blend_into_patch, lookup=None):
        # note that src = lookup is the current algorithm policy when lookup is not provided.
        # this policy could change in the future, so do not apply it here too despite being idempotent.
        h, w = src.shape[1:3]
        blk_size_upper_bound = block_size_upper_bound_for_seamless(ori, h, overlap)
        block_sizes = get_block_sizes(src, block_size, blk_size_upper_bound)
        rng: numpy.random.Generator = np.random.default_rng(seed=seed)

        lookup_batch_size = lookup.shape[0] if lookup is not None else 0
        finish_event, t, shm_name, shm_jobs = setup_pbar_seamless(
            ori, block_sizes, overlap, h, w, max(src.shape[0], lookup_batch_size))

        try:
            func = SeamlessMPFuncWrapper(
                False, ori, overlap, rng, blend_into_patch, version, shm_name, block_sizes, tolerance)

            is_batch = src.shape[0] > 1 or lookup_batch_size > 1
            output = batch_seamless_using_jobs(func, src, lookup, is_latent=False) if is_batch else \
                unwrap_latent_and_quilt_seamless(func, src, lookup, 0)
        finally:
            terminate_task(finish_event, shm_jobs, t)
        return (output,)


class ImageMakeSeamlessSB:
    """Transition stripe is built via a single rectangular block."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = get_quilting_shared_input_types()
        inputs.pop("parallelization_lvl")
        inputs.pop("tolerance")
        inputs.pop("seed")
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

    def compute(self, src, ori, block_size, overlap, version, blend_into_patch, lookup=None):
        # note that src = lookup is the current algorithm policy when lookup is not provided.
        # this policy could change in the future, so do not apply it here too despite being idempotent.
        h, _ = src.shape[1:3]
        blk_size_upper_bound = block_size_upper_bound_for_seamless(ori, h, overlap)
        block_sizes = get_block_sizes(src, block_size, blk_size_upper_bound)
        rng: numpy.random.Generator = np.random.default_rng(seed=0)

        lookup_batch_size = lookup.shape[0] if lookup is not None else 0
        finish_event, t, shm_name, shm_jobs = setup_pbar_seamless_v2(ori, max(src.shape[0], lookup_batch_size))

        try:
            func = SeamlessSPFuncWrapper(
                False, ori, overlap, rng, blend_into_patch, version, shm_name, block_sizes)

            is_batch = src.shape[0] > 1 or lookup_batch_size > 1
            output = batch_seamless_using_jobs(func, src, lookup, is_latent=False) if is_batch else \
                unwrap_latent_and_quilt_seamless(func, src, lookup, 0)
        finally:
            terminate_task(finish_event, shm_jobs, t)
        return (output,)


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


class LatentMakeSeamlessMB:
    """Transition stripe is built using overlapping square patches."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = LatentQuilting.INPUT_TYPES()["required"]  #get_quilting_shared_input_types()
        for to_remove in ["parallelization_lvl", "scale", "src"]:
            inputs.pop(to_remove)
        inputs["version"][1]["min"] = 1
        inputs["overlap"][1]["max"] = .5
        return {
            "required": {
                "src": ("LATENT",),
                "ori": (SEAMLESS_DIRS, {"default": SEAMLESS_DIRS[0]}),
                **inputs,
            },
            "optional": {
                "lookup": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "compute"
    CATEGORY = NODES_CATEGORY

    def compute(self, src, ori, block_size, overlap, tolerance, seed, version, blend_into_patch, lookup=None):
        src = src["samples"]
        lookup = lookup["samples"] if lookup is not None else None
        h, w = src.shape[2:4]
        rng: numpy.random.Generator = np.random.default_rng(seed=seed)

        lookup_batch_size = lookup.shape[0] if lookup is not None else 0
        finish_event, t, shm_name, shm_jobs = setup_pbar_seamless(
            ori, [block_size] * src.shape[0], overlap, h, w, max(src.shape[0], lookup_batch_size))

        try:
            func = SeamlessMPFuncWrapper(
                True, ori, overlap, rng, blend_into_patch, version, shm_name, [block_size], tolerance)

            is_batch = src.shape[0] > 1 or lookup_batch_size > 1
            output = batch_seamless_using_jobs(func, src, lookup, is_latent=True) if is_batch else \
                unwrap_latent_and_quilt_seamless(func, src, lookup, 0, is_latent=True)
        finally:
            terminate_task(finish_event, shm_jobs, t)
        return ({"samples": output},)


class LatentMakeSeamlessSB:
    """Transition stripe is built using overlapping square patches."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = LatentQuilting.INPUT_TYPES()["required"]
        for to_remove in ["tolerance", "parallelization_lvl", "scale", "src", "seed"]:
            inputs.pop(to_remove)
        inputs["version"][1]["min"] = 1
        inputs["overlap"][1]["max"] = .5
        return {
            "required": {
                "src": ("LATENT",),
                "ori": (SEAMLESS_DIRS, {"default": SEAMLESS_DIRS[0]}),
                **inputs,
            },
            "optional": {
                "lookup": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "compute"
    CATEGORY = NODES_CATEGORY

    def compute(self, src, ori, block_size, overlap, version, blend_into_patch, lookup=None):
        src = src["samples"]
        lookup = lookup["samples"] if lookup is not None else None
        rng: numpy.random.Generator = np.random.default_rng(seed=0)

        lookup_batch_size = lookup.shape[0] if lookup is not None else 0
        finish_event, t, shm_name, shm_jobs = setup_pbar_seamless_v2(ori, max(src.shape[0], lookup_batch_size))

        try:
            func = SeamlessSPFuncWrapper(True, ori, overlap, rng, blend_into_patch, version, shm_name, [block_size])

            is_batch = src.shape[0] > 1 or lookup_batch_size > 1
            output = batch_seamless_using_jobs(func, src, lookup, is_latent=True) if is_batch else \
                unwrap_latent_and_quilt_seamless(func, src, lookup, 0, is_latent=True)
        finally:
            terminate_task(finish_event, shm_jobs, t)
        return ({"samples": output},)

# endregion NODES
