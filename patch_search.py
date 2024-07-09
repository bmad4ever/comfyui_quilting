import numpy as np
import cv2 as cv
from .jena2020.generate import findPatchVertical, findPatchHorizontal, findPatchBoth

epsilon = np.finfo(float).eps


# region   get methods by version


def get_find_patch_to_the_right_method(version: int):
    match version:
        case 0:
            return findPatchHorizontal
        case 1:
            def v1_right(left_block, image, block_size, overlap, tolerance, rng):
                return find_patch_v1(left_block, None, None, None, image, block_size, overlap, tolerance, rng)

            return v1_right
        case 2:
            def v2_right(left_block, image, block_size, overlap, tolerance, rng):
                return find_patch_v2(left_block, None, None, None, image, block_size, overlap, tolerance, rng)

            return v2_right
        case 3:
            def v3_right(left_block, image, block_size, overlap, tolerance, rng):
                return find_patch_v3(left_block, None, None, None, image, block_size, overlap, tolerance, rng)

            return v3_right


def get_find_patch_below_method(version: int):
    match version:
        case 0:
            return findPatchVertical
        case 1:
            def v1_below(top_block, image, block_size, overlap, tolerance, rng):
                return find_patch_v1(None, None, top_block, None, image, block_size, overlap, tolerance, rng)

            return v1_below
        case 2:
            def v2_below(top_block, image, block_size, overlap, tolerance, rng):
                return find_patch_v2(None, None, top_block, None, image, block_size, overlap, tolerance, rng)

            return v2_below
        case 3:
            def v3_below(top_block, image, block_size, overlap, tolerance, rng):
                return find_patch_v3(None, None, top_block, None, image, block_size, overlap, tolerance, rng)

            return v3_below


def get_find_patch_both_method(version: int):
    match version:
        case 0:
            return findPatchBoth
        case 1:
            def v1_both(left_block, top_block, image, block_size, overlap, tolerance, rng):
                return find_patch_v1(left_block, None, top_block, None, image, block_size, overlap, tolerance, rng)

            return v1_both
        case 2:
            def v2_both(left_block, top_block, image, block_size, overlap, tolerance, rng):
                return find_patch_v2(left_block, None, top_block, None, image, block_size, overlap, tolerance, rng)

            return v2_both
        case 3:
            def v3_both(left_block, top_block, image, block_size, overlap, tolerance, rng):
                return find_patch_v3(left_block, None, top_block, None, image, block_size, overlap, tolerance, rng)

            return v3_both

# endregion


def find_patch_v1(ref_block_left, ref_block_right, ref_block_top, ref_block_bottom,
                  texture, block_size, overlap, tolerance,
                  rng: np.random.Generator
                  ):
    """
    Re-implementation of the version 1.0 solution using matchTemplate to improve performance.
    Uses the total instead of the mean for the errors matrix; other than that should be exactly the same.
    Does not output the same as version 1.0.
    """
    blks_sqdiffs = []
    if ref_block_left is not None:
        blks_sqdiffs.append(cv.matchTemplate(
            image=texture[:, :-block_size + overlap],
            templ=ref_block_left[:, -overlap:], method=cv.TM_SQDIFF))
    if ref_block_right is not None:
        blks_sqdiffs.append(cv.matchTemplate(
            image=np.roll(texture, -block_size + overlap, axis=1)[:, :-block_size + overlap],
            templ=ref_block_right[:, :overlap], method=cv.TM_SQDIFF))
    if ref_block_top is not None:
        blks_sqdiffs.append(cv.matchTemplate(
            image=texture[:-block_size + overlap, :],
            templ=ref_block_top[-overlap:, :], method=cv.TM_SQDIFF))
    if ref_block_bottom is not None:
        blks_sqdiffs.append(cv.matchTemplate(
            image=np.roll(texture, -block_size + overlap, axis=0)[:-block_size + overlap, :],
            templ=ref_block_bottom[:overlap, :], method=cv.TM_SQDIFF))

    err_mat = np.add.reduce(blks_sqdiffs)
    min_val = np.min(err_mat[err_mat > 0 if tolerance > 0 else True])  # ignore zeroes to enforce tolerance usage
    y, x = np.nonzero(err_mat <= (1.0 + tolerance) * min_val)
    c = rng.integers(len(y))
    y, x = y[c], x[c]
    return texture[y:y + block_size, x:x + block_size]


def find_patch_v2(ref_block_left, ref_block_right, ref_block_top, ref_block_bottom,
                  texture, block_size, overlap, tolerance,
                  rng: np.random.Generator
                  ):
    """
    Same as find_patch_v1 but chooses maximum error instead of the sum of errors,
        when patching with multiple adjacent blocks.
    Should use Lab color format (set via node before starting generation)
    """
    blks_sqdiffs = []
    if ref_block_left is not None:
        blks_sqdiffs.append(cv.matchTemplate(
            image=texture[:, :-block_size + overlap],
            templ=ref_block_left[:, -overlap:], method=cv.TM_SQDIFF))
    if ref_block_right is not None:
        blks_sqdiffs.append(cv.matchTemplate(
            image=np.roll(texture, -block_size + overlap, axis=1)[:, :-block_size + overlap],
            templ=ref_block_right[:, :overlap], method=cv.TM_SQDIFF))
    if ref_block_top is not None:
        blks_sqdiffs.append(cv.matchTemplate(
            image=texture[:-block_size + overlap, :],
            templ=ref_block_top[-overlap:, :], method=cv.TM_SQDIFF))
    if ref_block_bottom is not None:
        blks_sqdiffs.append(cv.matchTemplate(
            image=np.roll(texture, -block_size + overlap, axis=0)[:-block_size + overlap, :],
            templ=ref_block_bottom[:overlap, :], method=cv.TM_SQDIFF))

    err_mat = np.maximum.reduce(blks_sqdiffs)  # choose error from worst patch
    min_val = np.min(err_mat[err_mat > 0 if tolerance > 0 else True])  # ignore zeroes to enforce tolerance usage
    y, x = np.nonzero(err_mat <= (1.0 + tolerance) * min_val)
    c = rng.integers(len(y))
    y, x = y[c], x[c]
    return texture[y:y + block_size, x:x + block_size]


def find_patch_v3(ref_block_left, ref_block_right, ref_block_top, ref_block_bottom,
                  texture, block_size, overlap, tolerance,
                  rng: np.random.Generator
                  ):
    """
    This version makes use of TM_CCOEFF_NORMED in matchTemplate instead of TM_SQDIFF.
    """
    blks_ccs = []
    if ref_block_left is not None:
        blks_ccs.append(cv.matchTemplate(
            image=texture[:, :-block_size + overlap],
            templ=ref_block_left[:, -overlap:], method=cv.TM_CCOEFF_NORMED))
    if ref_block_right is not None:
        blks_ccs.append(cv.matchTemplate(
            image=np.roll(texture, -block_size + overlap, axis=1)[:, :-block_size + overlap],
            templ=ref_block_right[:, :overlap], method=cv.TM_CCOEFF_NORMED))
    if ref_block_top is not None:
        blks_ccs.append(cv.matchTemplate(
            image=texture[:-block_size + overlap, :],
            templ=ref_block_top[-overlap:, :], method=cv.TM_CCOEFF_NORMED))
    if ref_block_bottom is not None:
        blks_ccs.append(cv.matchTemplate(
            image=np.roll(texture, -block_size + overlap, axis=0)[:-block_size + overlap, :],
            templ=ref_block_bottom[:overlap, :], method=cv.TM_CCOEFF_NORMED))

    err_mat = 1 - np.minimum.reduce(blks_ccs)  # values from 0 to 2
    min_val = np.min(err_mat[err_mat > 0 if tolerance > 0 else True])  # ignore zeroes to enforce tolerance usage
    y, x = np.nonzero(err_mat <= (1.0 + tolerance) * min_val)
    c = rng.integers(len(y))
    y, x = y[c], x[c]
    return texture[y:y + block_size, x:x + block_size]