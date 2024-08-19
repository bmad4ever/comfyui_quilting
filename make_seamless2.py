# An alternative approach to making the texture seamless
from .synthesis_subroutines import compute_errors, get_match_template_method, get_min_cut_patch_horizontal
from .make_seamless import patch_horizontal_seam
from .types import UiCoordData
import numpy as np
import cv2 as cv


def seamless_horizontal(image, block_size, overlap, version, lookup_texture, rng,
                        uicd: UiCoordData | None = None):
    lookup_texture = image if lookup_texture is None else lookup_texture
    image = np.roll(image, +block_size // 2, axis=1)  # move seam to addressable space

    # left & right overlap errors
    template_method = get_match_template_method(version)
    lo_errs = cv.matchTemplate(image=lookup_texture[:, :-block_size],
                               templ=image[:, :overlap], method=template_method)
    if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(1):
        return None

    ro_errs = cv.matchTemplate(image=np.roll(lookup_texture, -block_size + overlap, axis=1)[:, :-block_size],
                               templ=image[:, block_size - overlap:block_size], method=template_method)
    if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(1):
        return None

    err_mat = compute_errors([lo_errs, ro_errs], version)
    min_val = np.min(err_mat)  # ignore tolerance in this solution
    y, x = np.nonzero(err_mat <= min_val)  # ignore tolerance here, choose only from the best values
    # still select randomly, it may be the case that there are more than one equally good matches
    # likely super rare, but doesn't costly to keep the option if eventually applicable
    c = rng.integers(len(y))
    y, x = y[c], x[c]

    # "fake" block will only contain the overlap, in order to re-use existing function.
    fake_left_block = np.empty((image.shape[0], image.shape[0], image.shape[2]), dtype=image.dtype)
    fake_right_block = np.empty((image.shape[0], image.shape[0], image.shape[2]), dtype=image.dtype)
    fake_left_block[:, -overlap:] = image[:, :overlap]
    fake_right_block[:, :overlap] = image[:, block_size - overlap:block_size]
    fake_block_sized_patch = np.empty((image.shape[0], image.shape[0], image.shape[2]), dtype=image.dtype)
    fake_block_sized_patch[:, :overlap] = lookup_texture[y:y + image.shape[0], x:x + overlap]
    fake_block_sized_patch[:, -overlap:] = lookup_texture[y:y + image.shape[0], x + block_size - overlap:x + block_size]
    left_side_patch = get_min_cut_patch_horizontal(fake_left_block, fake_block_sized_patch, image.shape[0], overlap)
    right_side_patch = np.fliplr(
        get_min_cut_patch_horizontal(
            np.fliplr(fake_right_block),
            np.fliplr(fake_block_sized_patch),
            image.shape[0], overlap
        )
    )
    if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(1):
        return None

    # paste vertical stripe patch
    image[:, :block_size] = lookup_texture[y:y + image.shape[0], x:x + block_size]
    image[:, :overlap] = left_side_patch[:, :overlap]
    image[:, block_size - overlap:block_size] = right_side_patch[:, -overlap:]

    return image


def seamless_vertical(image, block_size, overlap, version, lookup_texture, rng,
                      uicd: UiCoordData | None = None):
    rotated_solution = seamless_horizontal(np.rot90(image), block_size, overlap, version=version, rng=rng, uicd=uicd,
                                           lookup_texture=None if lookup_texture is None else np.rot90(lookup_texture))
    return np.rot90(rotated_solution, -1).copy()


def seamless_both(image, block_size, overlap, version, lookup_texture, rng,
                  uicd: UiCoordData | None = None):
    lookup_texture = image if lookup_texture is None else lookup_texture

    texture = seamless_vertical(image, block_size, overlap, version, lookup_texture, rng, uicd)
    if texture is None:
        return None
    texture = np.roll(texture, -block_size // 2, axis=0)  # center future seam at stripes interception
    texture = seamless_horizontal(texture, block_size, overlap, version, lookup_texture, rng, uicd)
    if texture is None:
        return None

    # center seam & patch it
    texture = np.roll(texture, texture.shape[0] // 2, axis=0)
    texture = np.roll(texture, texture.shape[1] // 2 - block_size // 2, axis=1)
    texture = patch_horizontal_seam(texture, lookup_texture, block_size, overlap, 0, rng)

    return texture
