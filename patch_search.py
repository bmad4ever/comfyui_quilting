from functools import lru_cache
from math import ceil
import numpy as np
import cv2 as cv

from .jena2020.generate import findPatchVertical, findPatchHorizontal, findPatchBoth
from .misc.bse_type_aliases import num_pixels

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
            raise NotImplementedError("Specified patch search version is not implemented.")


def get_match_template_method(version: int) -> int:
    match version:
        case 1:
            return cv.TM_SQDIFF
        case 2:
            return cv.TM_SQDIFF
        case 3:
            return cv.TM_CCOEFF_NORMED
        case _:
            raise NotImplementedError("Specified patch search version is not implemented.")


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
    if tolerance > 0:
        # attempt to ignore zeroes in order to apply tolerance, but mind edge case (e.g., blank image)
        min_val = np.min(pos_vals) if (pos_vals := err_mat[err_mat > 0]).size > 0 else 0
    else:
        min_val = np.min(err_mat)
    y, x = np.nonzero(err_mat <= (1.0 + tolerance) * min_val)
    c = rng.integers(len(y))
    y, x = y[c], x[c]
    return texture[y:y + block_size, x:x + block_size]


# min path cut 4 way should go to here?

# TODO because it is not a class func the cache must be cleared when a node finishes running!
@lru_cache(maxsize=4)
def patch_blending_vignette(block_size: num_pixels, overlap: num_pixels, left: bool, right: bool, top: bool, bottom: bool):
    margin = 1 #ceil(overlap / 12)  # must be small
    power = 2  # controls drop-off
    p = 6  # controls the shape

    def corner_distance(y, x):
        distance = ((abs(x - overlap + margin / 2) ** p + abs(y - overlap + margin / 2) ** p) ** (1 / p)) / (
                overlap - margin)
        return np.clip(distance, 0, 1)

    mask = np.ones((block_size, block_size), dtype=float)
    i, j = np.meshgrid(np.arange(overlap), np.arange(overlap))
    curve_top_left_corner = 1 - corner_distance(i, j) ** power

    # Corners
    # Top left corner
    if top and left:
        mask[:overlap, :overlap] = curve_top_left_corner
    elif top:
        mask[:overlap, :overlap] = curve_top_left_corner[:, -1].reshape(-1, 1)  # Copy the last column to all columns
    elif left:
        mask[:overlap, :overlap] = curve_top_left_corner[-1, :].reshape(1, -1)  # Copy the last row to all rows

    # Top right corner
    if top and right:
        mask[:overlap, -overlap:] = np.flip(curve_top_left_corner, axis=1)
    elif top:
        mask[:overlap, -overlap:] = curve_top_left_corner[:, -1].reshape(-1, 1)  # Copy the last column flipped
    elif right:
        mask[:overlap, -overlap:] = np.flip(curve_top_left_corner[-1, :]).reshape(1, -1)  # Copy the last row flipped

    # Bottom left corner
    if bottom and left:
        mask[-overlap:, :overlap] = np.flip(curve_top_left_corner, axis=0)
    elif bottom:
        mask[-overlap:, :overlap] = np.flip(curve_top_left_corner[:, -1]).reshape(-1, 1)  # Copy the last column flipped
    elif left:
        mask[-overlap:, :overlap] = curve_top_left_corner[-1, :]  # Copy the last row flipped

    # Bottom right corner
    if bottom and right:
        mask[-overlap:, -overlap:] = np.flip(curve_top_left_corner)
    elif bottom:
        mask[-overlap:, -overlap:] = (np.flip(curve_top_left_corner[:, -1])
                                      .reshape(-1, 1))  # Copy the last column flipped
    elif right:
        mask[-overlap:, -overlap:] = np.flip(curve_top_left_corner[-1, :]).reshape(1, -1)  # Copy the last row flipped

    # Edges
    if top:
        mask[:overlap, overlap:block_size - overlap] = (curve_top_left_corner[:, -1]
                                                        .reshape(-1, 1))  # Copy the last column vertically
    if bottom:
        mask[-overlap:, overlap:block_size - overlap] = (np.flip(curve_top_left_corner[:, -1])
                                                         .reshape(-1, 1))  # Copy the last column flipped vertically
    if left:
        mask[overlap:block_size - overlap, :overlap] = (curve_top_left_corner[-1, :]
                                                        .reshape(1, -1))  # Copy the last row horizontally
    if right:
        mask[overlap:block_size - overlap, -overlap:] = (np.flip(curve_top_left_corner[-1, :])
                                                         .reshape(1, -1))  # Copy the last row flipped horizontally
    return mask


def blur_patch_mask(src_mask, block_size: num_pixels, overlap: num_pixels, left: bool, right: bool, top: bool, bottom: bool):
    return src_mask  # don't use it for now until further testing

    vignette = patch_blending_vignette(block_size, overlap, left, right, top, bottom)

    # blurred = cv.GaussianBlur(src_mask, blur_ksize, blur_ksize)
    src_mask_uint = np.uint8(src_mask[:, :, 0] * 255)
    blurred = cv.distanceTransform(src_mask_uint, cv.DIST_L2, maskSize=0)
    blurred = cv.morphologyEx(blurred, cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)), iterations=1)
    blurred = cv.blur(blurred, (5, 5))
    blurred *= 255 / max(np.max(blurred), 1)
    blurred = np.float32(blurred / 255)

    #blurred = np.stack((blurred,) * src_mask.shape[2], axis=-1)
    #vignette = np.stack((vignette,) * src_mask.shape[2], axis=-1)

    result = (vignette * blurred) + ((1 - vignette) * src_mask[:, :, 0])
    result = np.stack((result, ) * src_mask.shape[2], axis=-1)
    return result
