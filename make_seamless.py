from .patch_search import get_generic_find_patch_method, blur_patch_mask
from .misc.bse_type_aliases import num_pixels
from .types import UiCoordData
from math import ceil
import numpy as np

inf = float('inf')


def get_numb_of_blocks_to_fill_stripe(block_size, overlap, dim_length):
    return int(ceil((dim_length - block_size) / (block_size - overlap)))


def make_seamless_horizontally(image, block_size, overlap, tolerance, rng: np.random.Generator,
                               version: int = 1, lookup_texture=None, uicd: UiCoordData | None = None):
    """
    @param image: the image to make seamless; also be used to fetch the patches.
    @param fnc: when evaluating potential patches the errors of different adjacency will be combined using this function
    @param lookup_texture: if provided, the patches will be obtained from "lookup_texture" instead.
    """
    lookup_texture = image if lookup_texture is None else lookup_texture

    bmo = block_size - overlap

    src_h, src_w = image.shape[:2]
    n_h = get_numb_of_blocks_to_fill_stripe(block_size, overlap, src_h)  #int(ceil((src_h - block_size) / bmo))

    texture_map = np.zeros((src_h, src_w, image.shape[-1])).astype(image.dtype)
    texture_map[:src_h, :src_w] = image

    # roll texture map to allow overlapping between left and right blocks.
    # this allows for big block sizes, and, consequently, for faster generations
    left_blocks = np.roll(texture_map, block_size // 2 - overlap + block_size, axis=1)[:, :block_size]
    right_blocks = np.roll(texture_map, -round(block_size / 2) + overlap, axis=1)[:, :block_size]

    # center v seam at half block distance of the left corner
    texture_map = np.roll(texture_map, block_size // 2, axis=1)

    # get 1st patch
    ref_block_left = left_blocks[:block_size, :block_size]
    ref_block_right = right_blocks[:block_size, :block_size]

    find_patch = get_generic_find_patch_method(version=version)

    patch_block = find_patch(
        ref_block_left, ref_block_right, None, None, lookup_texture, block_size, overlap, tolerance, rng)  #, fnc)
    min_cut_patch = get_4way_min_cut_patch(ref_block_left, ref_block_right, None, None,
                                           patch_block, block_size, overlap)
    texture_map[:block_size, :block_size] = min_cut_patch
    if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(1):
        return None

    for y in range(1, n_h):
        blk_1y = y * bmo  # block top corner y
        blk_2y = blk_1y + block_size  # block bottom corner y

        # get adjacent blocks
        ref_block_left = left_blocks[blk_1y:blk_2y, :block_size]
        ref_block_right = right_blocks[blk_1y:blk_2y, :block_size]
        ref_block_top = texture_map[(blk_1y - bmo):(blk_1y + overlap), :block_size]

        patch_block = find_patch(ref_block_left, ref_block_right, ref_block_top, None,
                                 lookup_texture, block_size, overlap, tolerance, rng)
        min_cut_patch = get_4way_min_cut_patch(ref_block_left, ref_block_right, ref_block_top, None,
                                               patch_block, block_size, overlap)

        texture_map[blk_1y:blk_2y, :block_size] = min_cut_patch
        if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(1):
            return None

    # fill last block
    ref_block_left = left_blocks[-block_size:, :block_size]
    ref_block_right = right_blocks[-block_size:, :block_size]
    ref_block_top = np.empty_like(ref_block_left)  # only copy overlap
    ref_block_top[-overlap:, :] = texture_map[-block_size:-block_size + overlap, :block_size]
    patch_block = find_patch(ref_block_left, ref_block_right, ref_block_top, None,
                             lookup_texture, block_size, overlap, tolerance, rng)
    min_cut_patch = get_4way_min_cut_patch(ref_block_left, ref_block_right, ref_block_top, None,
                                           patch_block, block_size, overlap)
    texture_map[-block_size:, :block_size] = min_cut_patch
    if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(1):
        return None

    return texture_map


def get_min_cut_patch_mask_horizontal(block1, block2, block_size: num_pixels, overlap: num_pixels):
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
    mask = np.zeros((block_size, block_size, block1.shape[2]), dtype=block1.dtype)
    for i in range(len(path)):
        mask[i, :path[i] + 1] = 1
    return mask


def get_4way_min_cut_patch(ref_block_left, ref_block_right, ref_block_top, ref_block_bottom,
                           patch_block, block_size, overlap):
    # (optional step) blur masks for a more seamless integration ( sometimes makes transition more noticeable, depends )
    masks_list = []

    has_left = ref_block_left is None
    has_right = ref_block_right is not None
    has_top = ref_block_top is not None
    has_bottom = ref_block_bottom is not None

    if has_left:
        mask_left = get_min_cut_patch_mask_horizontal(ref_block_left, patch_block, block_size, overlap)
        mask_left = blur_patch_mask(mask_left, block_size, overlap, has_left, has_right, has_top, has_bottom)
        masks_list.append(mask_left)

    if has_right:
        mask_right = get_min_cut_patch_mask_horizontal(np.fliplr(ref_block_right), np.fliplr(patch_block), block_size,
                                                       overlap)
        mask_right = np.fliplr(mask_right)
        mask_right = blur_patch_mask(mask_right, block_size, overlap, has_left, has_right, has_top, has_bottom)
        masks_list.append(mask_right)

    if has_top:
        # V , >  counterclockwise rotation
        mask_top = get_min_cut_patch_mask_horizontal(np.rot90(ref_block_top), np.rot90(patch_block), block_size, overlap)
        mask_top = np.rot90(mask_top, 3)
        mask_top = blur_patch_mask(mask_top, block_size, overlap, has_left, has_right, has_top, has_bottom)
        masks_list.append(mask_top)

    if has_bottom:
        mask_bottom = get_min_cut_patch_mask_horizontal(np.fliplr(np.rot90(ref_block_bottom)),
                                                        np.fliplr(np.rot90(patch_block)), block_size, overlap)
        mask_bottom = np.rot90(np.fliplr(mask_bottom), 3)
        mask_bottom = blur_patch_mask(mask_bottom, block_size, overlap, has_left, has_right, has_top, has_bottom)
        masks_list.append(mask_bottom)

    # --- apply masks and return block ---
    # compute auxiliary data
    mask_s = sum(masks_list)
    masks_max = np.maximum.reduce(masks_list)
    mask_mos = np.divide(masks_max, mask_s, out=np.zeros_like(mask_s), where=mask_s != 0)
    # note -> if blurred, masks weights are scaled with respect the max mask value.
    # example: mask1: 0.2 mask2:0.5 -> max = .5 ; sum = .7 -> (.2*lb + .5*tb) * (max/sum) + patch * (1 - max)

    # place adjacent block sections
    res_block = np.zeros_like(patch_block)
    if has_left:
        res_block += mask_left * np.roll(ref_block_left, overlap, 1)
    if has_right:
        res_block += mask_right * np.roll(ref_block_right, -overlap, 1)
    if has_top:
        res_block += mask_top * np.roll(ref_block_top, overlap, 0)
    if has_bottom:
        res_block += mask_bottom * np.roll(ref_block_bottom, -overlap, 0)
    res_block *= mask_mos

    # place patch section
    patch_weight = 1 - masks_max
    res_block = res_block + patch_weight * patch_block

    return res_block


def make_seamless_vertically(image, block_size, overlap, tolerance, rng: np.random.Generator,
                             version: int = 1, lookup_texture=None, uicd: UiCoordData | None = None):
    rotated_solution = make_seamless_horizontally(
        np.rot90(image, 1), block_size, overlap, tolerance, rng=rng, version=version, uicd=uicd,
        lookup_texture=None if lookup_texture is None else np.rot90(lookup_texture))
    return np.rot90(rotated_solution, -1).copy() if rotated_solution is not None else None


def make_seamless_both(image, block_size, overlap, tolerance, rng: np.random.Generator,
                       version: int = 1, lookup_texture=None, uicd: UiCoordData | None = None):
    lookup_texture = image if lookup_texture is None else lookup_texture

    # patch the texture in both directions. the last stripe's endpoints won't loop yet.
    texture = make_seamless_vertically(image, block_size, overlap, tolerance, rng,
                                       version=version, lookup_texture=lookup_texture, uicd=uicd)
    if texture is not None:
        texture = np.roll(texture, -block_size // 2, axis=0)  # center future seam at stripes interception
        texture = make_seamless_horizontally(texture, block_size, overlap, tolerance, rng,
                                             version=version, lookup_texture=lookup_texture, uicd=uicd)
    if texture is None:
        return None

    #   center the area to patch 1st, this will make the rolls in the next step easier
    texture = np.roll(texture, texture.shape[0] // 2, axis=0)
    texture = np.roll(texture, (texture.shape[1] - block_size) // 2, axis=1)

    return patch_horizontal_seam(texture, lookup_texture, block_size, overlap, tolerance, rng,
                                 version=version, uicd=uicd)


def patch_horizontal_seam(texture_to_patch, lookup_texture, block_size, overlap, tolerance,
                          rng: np.random.Generator, version: int = 1, uicd: UiCoordData | None = None):
    """
    Patches the center of the texture
    """
    ys = (texture_to_patch.shape[0] - block_size) // 2
    ye = ys + block_size
    xs = (texture_to_patch.shape[1] - block_size) // 2
    xe = xs + block_size

    find_patch = get_generic_find_patch_method(version=version)

    # PATCH H SEAM -> LEFT PATCH
    adj_top_blk = np.roll(texture_to_patch, ye - overlap, axis=0)[-block_size:, xs - overlap:xe - overlap]
    adj_btm_blk = np.roll(texture_to_patch, -ye + overlap, axis=0)[:block_size, xs - overlap:xe - overlap]
    adj_lft_blk = np.roll(texture_to_patch, -xs, axis=1)[ys:ye, -block_size:]
    patch = find_patch(adj_lft_blk, None, adj_top_blk, adj_btm_blk,
                       lookup_texture, block_size, overlap, tolerance, rng)
    patch = get_4way_min_cut_patch(adj_lft_blk, None, adj_top_blk, adj_btm_blk,
                                   patch, block_size, overlap)
    texture_to_patch[ys:ye, xs - overlap:xe - overlap] = patch
    if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(1):
        return None

    # PATCH H SEAM -> RIGHT PATCH
    adj_top_blk = np.roll(texture_to_patch, ye - overlap, axis=0)[-block_size:, xs + overlap:xe + overlap]
    adj_btm_blk = np.roll(texture_to_patch, -ye + overlap, axis=0)[:block_size, xs + overlap:xe + overlap]
    adj_lft_blk = np.roll(texture_to_patch, -xs - overlap * 2, axis=1)[ys:ye, -block_size:]  # review this one
    adj_rgt_blk = np.roll(texture_to_patch, -xs - block_size, axis=1)[ys:ye, :block_size]
    patch = find_patch(adj_lft_blk, adj_rgt_blk, adj_top_blk, adj_btm_blk,
                       lookup_texture, block_size, overlap, tolerance, rng)
    patch = get_4way_min_cut_patch(adj_lft_blk, adj_rgt_blk, adj_top_blk, adj_btm_blk,
                                   patch, block_size, overlap)
    texture_to_patch[ys:ye, xs + overlap:xe + overlap] = patch
    if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(1):
        return None

    return texture_to_patch
