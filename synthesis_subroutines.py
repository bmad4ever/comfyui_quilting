from functools import lru_cache
import importlib.util
import numpy as np
import cv2 as cv

from .jena2020.generate import (inf, findPatchHorizontal, findPatchVertical, findPatchBoth,
                                getMinCutPatchHorizontal, getMinCutPatchVertical, getMinCutPatchBoth)
from .types import GenParams, num_pixels

epsilon = np.finfo(float).eps


# region   get methods by version


def get_find_patch_to_the_right_method(version: int):
    match version:
        case 0:
            def jena_right(left, source, gen_args: GenParams, rng):
                return findPatchHorizontal(left, source, gen_args.block_size, gen_args.overlap, gen_args.tolerance, rng)

            return jena_right
        case _:
            def vx_right(left_block, image, gen_args, rng):
                return find_patch_vx(left_block, None, None, None, image, gen_args, rng)

            return vx_right


def get_find_patch_below_method(version: int):
    match version:
        case 0:
            def jena_below(top, source, gen_args: GenParams, rng):
                return findPatchVertical(top, source, gen_args.block_size, gen_args.overlap, gen_args.tolerance, rng)

            return jena_below
        case _:
            def vx_below(top_block, image, gen_args, rng):
                return find_patch_vx(None, None, top_block, None, image, gen_args, rng)

            return vx_below


def get_find_patch_both_method(version: int):
    match version:
        case 0:
            def jena_both(left, top, source, gen_args: GenParams, rng):
                return findPatchBoth(left, top, source, gen_args.block_size, gen_args.overlap, gen_args.tolerance, rng)

            return jena_both
        case _:
            def vx_both(left_block, top_block, image, gen_args, rng):
                return find_patch_vx(left_block, None, top_block, None, image, gen_args, rng)

            return vx_both


def get_min_cut_patch_horizontal_method(version: int):
    if version == 0:
        def jena_cut_h(left, patch, gen_args: GenParams):
            return getMinCutPatchHorizontal(left, patch, gen_args.block_size, gen_args.overlap)

        return jena_cut_h
    return get_min_cut_patch_horizontal


def get_min_cut_patch_vertical_method(version: int):
    if version == 0:
        def jena_cut_v(top, patch, gen_args: GenParams):
            return getMinCutPatchVertical(top, patch, gen_args.block_size, gen_args.overlap)

        return jena_cut_v
    return get_min_cut_patch_vertical


def get_min_cut_patch_both_method(version: int):
    if version == 0:
        def jena_cut_both(left, top, patch, gen_args: GenParams):
            return getMinCutPatchBoth(left, top, patch, gen_args.block_size, gen_args.overlap)

        return jena_cut_both
    return get_min_cut_patch_both


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


# region  custom implementation of: patch search & min cut + auxiliary methods

def find_patch_vx(ref_block_left, ref_block_right, ref_block_top, ref_block_bottom,
                  texture, gen_args: GenParams,
                  rng: np.random.Generator):
    block_size, overlap, tolerance = gen_args.bot
    blks_diffs = []
    template_method = get_match_template_method(gen_args.version)
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

    err_mat = compute_errors(blks_diffs, gen_args.version)
    if tolerance > 0:
        # attempt to ignore zeroes in order to apply tolerance, but mind edge case (e.g., blank image)
        min_val = np.min(pos_vals) if (pos_vals := err_mat[err_mat > 0]).size > 0 else 0
    else:
        min_val = np.min(err_mat)
    y, x = np.nonzero(err_mat <= (1.0 + tolerance) * min_val)
    c = rng.integers(len(y))
    y, x = y[c], x[c]
    return texture[y:y + block_size, x:x + block_size]


def get_4way_min_cut_patch(ref_block_left, ref_block_right, ref_block_top, ref_block_bottom,
                           patch_block, gen_args: GenParams):
    # note -> if blurred, masks weights are scaled with respect the max mask value.
    # example: mask1: 0.2 mask2:0.5 -> max = .5 ; sum = .7 -> (.2*lb + .5*tb) * (max/sum) + patch * (1 - max)
    # likely "heavier" to compute, but does not require handling each corner individually

    block_size, overlap = gen_args.bo

    masks_list = []

    has_left = ref_block_left is not None
    has_right = ref_block_right is not None
    has_top = ref_block_top is not None
    has_bottom = ref_block_bottom is not None

    res_block = np.zeros_like(patch_block)

    def process_block(rolled_block, mask):
        if gen_args.blend_into_patch:
            mask = blur_patch_mask(mask, block_size, overlap, has_left, has_right, has_top, has_bottom)
        masks_list.append(mask)
        np.add(apply_mask(rolled_block, mask, True), res_block, out=res_block)

    if has_left:
        mask_left = get_min_cut_patch_mask_horizontal(ref_block_left, patch_block, block_size, overlap)
        process_block(np.roll(ref_block_left, overlap, 1), mask_left)

    if has_right:
        mask_right = get_min_cut_patch_mask_horizontal(np.fliplr(ref_block_right), np.fliplr(patch_block),
                                                       block_size, overlap)
        mask_right = np.fliplr(mask_right)
        process_block(np.roll(ref_block_right, -overlap, 1), mask_right)

    if has_top:
        # V , >  counterclockwise rotation
        mask_top = get_min_cut_patch_mask_horizontal(np.rot90(ref_block_top), np.rot90(patch_block),
                                                     block_size, overlap)
        mask_top = np.rot90(mask_top, 3)
        process_block(np.roll(ref_block_top, overlap, 0), mask_top)

    if has_bottom:
        mask_bottom = get_min_cut_patch_mask_horizontal(np.fliplr(np.rot90(ref_block_bottom)),
                                                        np.fliplr(np.rot90(patch_block)), block_size, overlap)
        mask_bottom = np.rot90(np.fliplr(mask_bottom), 3)
        process_block(np.roll(ref_block_bottom, -overlap, 0), mask_bottom)

    # compute auxiliary data to weight original/patch & fix sum at the corners
    mask_s = sum(masks_list)
    masks_max = np.maximum.reduce(masks_list)
    mask_mos = np.divide(masks_max, mask_s, out=np.zeros_like(mask_s), where=mask_s != 0)

    apply_mask(res_block, mask_mos, True)  # weight res_block (also fixes sum at corners)
    patch_weight = np.subtract(1, masks_max, out=masks_max)  # note that masks_max is no longer needed
    return np.add(res_block, apply_mask(patch_block, patch_weight), out=res_block)


def apply_mask(src: np.ndarray, mask: np.ndarray, overwrite: bool = False):
    """
    @param src:  image or latent with shape of length 3 (height, width, channels)
    @param mask: mask with shape of length 2 (height, width)
    @param overwrite: overwrite src w/ the result
    """
    output = src if overwrite else np.empty_like(src)
    for c_i in range(src.shape[-1]):
        np.multiply(src[:, :, c_i], mask, out=output[:, :, c_i])
    return output


@lru_cache(maxsize=4)
def patch_blending_vignette(block_size: num_pixels, overlap: num_pixels,
                            left: bool, right: bool, top: bool, bottom: bool) -> np.ndarray:
    margin = 1  # must be small !
    power = 2.5  # controls drop-off
    p = 6  # controls the shape

    def corner_distance(y, x):
        distance = ((abs(x - overlap + margin / 2) ** p + abs(y - overlap + margin / 2) ** p) ** (1 / p)) / (
                overlap - margin)
        return np.clip(distance, 0, 1)

    mask = np.ones((block_size, block_size), dtype=np.float32)
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
        mask[:overlap, -overlap:] = curve_top_left_corner[:, -1].reshape(-1, 1)
    elif right:
        mask[:overlap, -overlap:] = np.flip(curve_top_left_corner[-1, :]).reshape(1, -1)

    # Bottom left corner
    if bottom and left:
        mask[-overlap:, :overlap] = np.flip(curve_top_left_corner, axis=0)
    elif bottom:
        mask[-overlap:, :overlap] = np.flip(curve_top_left_corner[:, -1]).reshape(-1, 1)
    elif left:
        mask[-overlap:, :overlap] = curve_top_left_corner[-1, :].reshape(1, -1)

    # Bottom right corner
    if bottom and right:
        mask[-overlap:, -overlap:] = np.flip(curve_top_left_corner)
    elif bottom:
        mask[-overlap:, -overlap:] = (np.flip(curve_top_left_corner[:, -1])
                                      .reshape(-1, 1))
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


def blur_patch_mask(src_mask, block_size: num_pixels, overlap: num_pixels, left: bool, right: bool, top: bool,
                    bottom: bool):
    vignette = patch_blending_vignette(block_size, overlap, left, right, top, bottom)

    # compute dst_grad & edge_blurred
    blurred_a = np.ascontiguousarray(src_mask * 255, dtype=np.uint8)
    blurred_b = np.empty_like(blurred_a, order='c')
    cv.morphologyEx(blurred_a, cv.MORPH_ERODE, iterations=1, dst=blurred_b,
                    kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))

    cv.blur(blurred_b, (3, 3), dst=blurred_a)
    edge_blurred = np.float32(blurred_a) / 255

    cv.distanceTransform(blurred_b, cv.DIST_L2, maskSize=0, dst=blurred_a, dstType=0)
    cv.blur(blurred_a, (5, 5), dst=blurred_b)
    dst_grad = np.float32(blurred_b) / max(np.max(blurred_b), 1)

    # compute formula re-using already allocated memory
    #   formula: (vignette * dst_grad) + ((1 - vignette) * edge_blurred)
    weighted_blurred = np.multiply(vignette, dst_grad, out=dst_grad)
    one_minus_vignette = 1 - vignette  # vignette is cached, can't edit it
    weighted_src = np.multiply(edge_blurred, one_minus_vignette, out=one_minus_vignette)
    result = np.add(weighted_blurred, weighted_src, out=weighted_blurred)
    return np.clip(result, 0, 1, out=result)  # better safe than sorry


def get_min_cut_patch_mask_horizontal_jena2020(block1, block2, block_size: num_pixels, overlap: num_pixels):
    """
    @param block1: block to the left, with the overlap on its right edge
    @param block2: block to the right, with the overlap on its left edge
    @return: ONLY the mask (not the patched overlap section)
    """
    err = ((block1[:, -overlap:] - block2[:, :overlap]) ** 2).mean(2)
    # maintain minIndex for 2nd row onwards and
    min_index = []
    E = [list(err[0])]
    for i in range(1, err.shape[0]):
        # Get min values and args, -1 = left, 0 = middle, 1 = right
        e = [inf] + E[-1] + [inf]
        e = np.array([e[:-2], e[1:-1], e[2:]])
        # Get minIndex
        min_arr = e.min(0)
        min_arg = e.argmin(0) - 1
        min_index.append(min_arg)
        # Set Eij = e_ij + min_
        Eij = err[i] + min_arr
        E.append(list(Eij))

    # Check the last element and backtrack to find path
    path = []
    min_arg = np.argmin(E[-1])
    path.append(min_arg)

    # Backtrack to min path
    for idx in min_index[::-1]:
        min_arg = min_arg + idx[min_arg]
        path.append(min_arg)
    # Reverse to find full path
    path = path[::-1]
    mask = np.zeros((block_size, block_size), dtype=block1.dtype)
    for i in range(len(path)):
        mask[i, :path[i] + 1] = 1
    return mask


if importlib.util.find_spec("pyastar2d") is not None:
    import pyastar2d


    def get_min_cut_patch_mask_horizontal_astar(block1, block2, block_size: num_pixels, overlap: num_pixels):
        """
        @param block1: block to the left, with the overlap on its right edge
        @param block2: block to the right, with the overlap on its left edge
        @return: ONLY the mask (not the patched overlap section)
        """
        err = ((block1[:, -overlap:] - block2[:, :overlap]) ** 2).mean(2)
        err *= block_size ** 3
        err += 1
        err *= block_size ** 3  # make the lowest value big enough for 1 to be negligible
        err = np.pad(err, ((1, 1), (0, 0)), 'constant', constant_values=(1, 1))

        start = (0, err.shape[1] // 2)
        end = (err.shape[0] - 1, err.shape[1] // 2)

        path = pyastar2d.astar_path(err, start, end, allow_diagonal=True)
        mask = np.ones((block_size, block_size), dtype=block1.dtype)
        shape_m2 = err.shape[0] - 2

        start_index = 0  # find start index to avoid checking 0 < i every iteration
        for idx, (i, j) in enumerate(path):
            if 0 < i:
                start_index = idx
                break

        for i, j in path[start_index:]:  # draw path
            mask[i - 1, j + 1] = 0
            if i >= shape_m2:
                break

        cv.floodFill(mask, None, (mask.shape[0] - 1, mask.shape[1] - 1), (0,))
        return mask


    get_min_cut_patch_mask_horizontal = get_min_cut_patch_mask_horizontal_astar
else:
    get_min_cut_patch_mask_horizontal = get_min_cut_patch_mask_horizontal_jena2020


# endregion


# region min cut patch aliases

def get_min_cut_patch_horizontal(left_block, patch_block, gen_args: GenParams):
    return get_4way_min_cut_patch(
        left_block, None, None, None,
        patch_block, gen_args
    )


def get_min_cut_patch_vertical(top_block, patch_block, gen_args: GenParams):
    return get_4way_min_cut_patch(
        None, None, top_block, None,
        patch_block, gen_args
    )


def get_min_cut_patch_both(left_block, top_block, patch_block, gen_args: GenParams):
    return get_4way_min_cut_patch(
        left_block, None, top_block, None,
        patch_block, gen_args
    )

# endregion
