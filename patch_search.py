from .jena2020.generate import findPatchVertical, findPatchHorizontal, findPatchBoth
import numpy as np
import cv2 as cv

epsilon = np.finfo(float).eps


# region   get methods by version


def get_find_patch_to_the_right_method(version: int):
    match version:
        case 0:
            return findPatchHorizontal
        case _:
            def vx_right(left_block, image, block_size, overlap, tolerance, rng):
                return find_patch_vx(left_block, None, None, None,
                                     image, block_size, overlap, tolerance, rng, version)

            return vx_right


def get_find_patch_below_method(version: int):
    match version:
        case 0:
            return findPatchVertical
        case _:
            def vx_below(top_block, image, block_size, overlap, tolerance, rng):
                return find_patch_vx(None, None, top_block, None,
                                     image, block_size, overlap, tolerance, rng, version)

            return vx_below


def get_find_patch_both_method(version: int):
    match version:
        case 0:
            return findPatchBoth
        case _:
            def vx_both(left_block, top_block, image, block_size, overlap, tolerance, rng):
                return find_patch_vx(left_block, None, top_block, None,
                                     image, block_size, overlap, tolerance, rng, version)

            return vx_both


def get_generic_find_patch_method(version: int):
    def vx_patch_find(ref_block_left, ref_block_right, ref_block_top, ref_block_bottom,
                      texture, block_size, overlap, tolerance, rng):
        return find_patch_vx(ref_block_left, ref_block_right, ref_block_top, ref_block_bottom,
                             texture, block_size, overlap, tolerance, rng, version)
    return vx_patch_find


def compute_errors(diffs: list[np.ndarray], version: int) -> np.ndarray:
    match version:
        case 1:
            return np.add.reduce(diffs)
        case 2:
            return np.maximum.reduce(diffs)
        case 3:
            return 1 - np.minimum.reduce(diffs)  # values from 0 to 2
        case _:
            raise NotImplemented()


def get_match_template_method(version: int) -> int:
    match version:
        case 1:
            return cv.TM_SQDIFF
        case 2:
            return cv.TM_SQDIFF
        case 3:
            return cv.TM_CCOEFF_NORMED
        case _:
            raise NotImplemented()


# endregion


def find_patch_vx(ref_block_left, ref_block_right, ref_block_top, ref_block_bottom,
                  texture, block_size, overlap, tolerance,
                  rng: np.random.Generator, version):
    blks_diffs = []
    template_method = get_match_template_method(version)
    if ref_block_left is not None:
        blks_diffs.append(cv.matchTemplate(
            image=texture[:, :-block_size + overlap],
            templ=ref_block_left[:, -overlap:], method=template_method))
    if ref_block_right is not None:
        blks_diffs.append(cv.matchTemplate(
            image=np.roll(texture, -block_size + overlap, axis=1)[:, :-block_size + overlap],
            templ=ref_block_right[:, :overlap], method=template_method))
    if ref_block_top is not None:
        blks_diffs.append(cv.matchTemplate(
            image=texture[:-block_size + overlap, :],
            templ=ref_block_top[-overlap:, :], method=template_method))
    if ref_block_bottom is not None:
        blks_diffs.append(cv.matchTemplate(
            image=np.roll(texture, -block_size + overlap, axis=0)[:-block_size + overlap, :],
            templ=ref_block_bottom[:overlap, :], method=template_method))

    err_mat = compute_errors(blks_diffs, version)
    min_val = np.min(err_mat[err_mat > 0 if tolerance > 0 else True])  # ignore zeroes to enforce tolerance usage
    y, x = np.nonzero(err_mat <= (1.0 + tolerance) * min_val)
    c = rng.integers(len(y))
    y, x = y[c], x[c]
    return texture[y:y + block_size, x:x + block_size]

