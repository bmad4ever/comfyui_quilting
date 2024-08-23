from .synthesis_subroutines import get_4way_min_cut_patch, find_patch_vx
from .types import UiCoordData, GenParams
from math import ceil
import numpy as np


def get_numb_of_blocks_to_fill_stripe(block_size, overlap, dim_length):
    return int(ceil((dim_length - block_size) / (block_size - overlap)))


def make_seamless_horizontally(image, gen_args: GenParams, rng: np.random.Generator, lookup_texture=None,
                               uicd: UiCoordData | None = None):
    """
    @param image: the image to make seamless; also be used to fetch the patches.
    @param fnc: when evaluating potential patches the errors of different adjacency will be combined using this function
    @param lookup_texture: if provided, the patches will be obtained from "lookup_texture" instead.
    """
    lookup_texture = image if lookup_texture is None else lookup_texture

    block_size, overlap, tolerance = gen_args.block_size, gen_args.overlap, gen_args.tolerance
    bmo = block_size - overlap

    src_h, src_w = image.shape[:2]
    n_h = get_numb_of_blocks_to_fill_stripe(block_size, overlap, src_h)

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

    patch_block = find_patch_vx(
        ref_block_left, ref_block_right, None, None, lookup_texture, gen_args, rng)
    min_cut_patch = get_4way_min_cut_patch(ref_block_left, ref_block_right, None, None,
                                           patch_block, gen_args)
    texture_map[:block_size, :block_size] = min_cut_patch
    if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(1):
        return None

    def fix_corners():
        ref_block_left[:overlap, -overlap:] = ref_block_top[-overlap:, :overlap]
        ref_block_right[:overlap, :overlap] = ref_block_top[-overlap:, -overlap:]

    for y in range(1, n_h):
        blk_1y = y * bmo  # block top corner y
        blk_2y = blk_1y + block_size  # block bottom corner y

        # get adjacent blocks
        ref_block_left = left_blocks[blk_1y:blk_2y, :block_size]
        ref_block_right = right_blocks[blk_1y:blk_2y, :block_size]
        ref_block_top = texture_map[(blk_1y - bmo):(blk_1y + overlap), :block_size]
        fix_corners()

        patch_block = find_patch_vx(ref_block_left, ref_block_right, ref_block_top, None,
                                    lookup_texture, gen_args, rng)
        min_cut_patch = get_4way_min_cut_patch(ref_block_left, ref_block_right, ref_block_top, None,
                                               patch_block, gen_args)

        texture_map[blk_1y:blk_2y, :block_size] = min_cut_patch
        if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(1):
            return None

    # fill last block
    ref_block_left = left_blocks[-block_size:, :block_size]
    ref_block_right = right_blocks[-block_size:, :block_size]
    ref_block_top = np.empty_like(ref_block_left)  # only copy overlap
    ref_block_top[-overlap:, :] = texture_map[-block_size:-block_size + overlap, :block_size]
    fix_corners()
    patch_block = find_patch_vx(ref_block_left, ref_block_right, ref_block_top, None,
                                lookup_texture, gen_args, rng)
    min_cut_patch = get_4way_min_cut_patch(ref_block_left, ref_block_right, ref_block_top, None,
                                           patch_block, gen_args)
    texture_map[-block_size:, :block_size] = min_cut_patch
    if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(1):
        return None

    return texture_map


def make_seamless_vertically(image, gen_args: GenParams, rng: np.random.Generator,
                             lookup_texture=None, uicd: UiCoordData | None = None):
    rotated_solution = make_seamless_horizontally(
        np.rot90(image, 1), gen_args, rng=rng, uicd=uicd,
        lookup_texture=None if lookup_texture is None else np.rot90(lookup_texture))
    return np.rot90(rotated_solution, -1).copy() if rotated_solution is not None else None


def make_seamless_both(image, gen_args: GenParams, rng: np.random.Generator,
                       lookup_texture=None, uicd: UiCoordData | None = None):
    lookup_texture = image if lookup_texture is None else lookup_texture
    block_size = gen_args.block_size

    # patch the texture in both directions. the last stripe's endpoints won't loop yet.
    texture = make_seamless_vertically(image, gen_args, rng, lookup_texture=lookup_texture, uicd=uicd)
    if texture is not None:
        texture = np.roll(texture, -block_size // 2, axis=0)  # center future seam at stripes interception
        texture = make_seamless_horizontally(texture, gen_args, rng, lookup_texture=lookup_texture, uicd=uicd)
    if texture is None:
        return None

    #   center the area to patch 1st, this will make the rolls in the next step easier
    texture = np.roll(texture, texture.shape[0] // 2, axis=0)
    texture = np.roll(texture, (texture.shape[1] - block_size) // 2, axis=1)

    return patch_horizontal_seam(texture, lookup_texture, gen_args, rng, uicd=uicd)


def patch_horizontal_seam(texture_to_patch, lookup_texture, gen_args: GenParams,
                          rng: np.random.Generator, uicd: UiCoordData | None = None):
    """
    Patches the center of the texture
    """
    block_size, overlap = gen_args.bo

    ys = (texture_to_patch.shape[0] - block_size) // 2
    ye = ys + block_size
    xs = (texture_to_patch.shape[1] - block_size) // 2
    xe = xs + block_size

    # PATCH H SEAM -> LEFT PATCH
    adj_top_blk = np.roll(texture_to_patch, -ys - overlap, axis=0)[-block_size:, xs - overlap:xe - overlap]
    adj_btm_blk = np.roll(texture_to_patch, -ye + overlap, axis=0)[:block_size, xs - overlap:xe - overlap]
    adj_lft_blk = np.roll(texture_to_patch, -xs, axis=1)[ys:ye, -block_size:]
    patch = find_patch_vx(adj_lft_blk, None, adj_top_blk, adj_btm_blk, lookup_texture, gen_args, rng)
    patch = get_4way_min_cut_patch(adj_lft_blk, None, adj_top_blk, adj_btm_blk, patch, gen_args)
    texture_to_patch[ys:ye, xs - overlap:xe - overlap] = patch
    if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(1):
        return None

    # PATCH H SEAM -> RIGHT PATCH
    adj_top_blk = np.roll(texture_to_patch, -ys - overlap, axis=0)[-block_size:, xs + overlap:xe + overlap]
    adj_btm_blk = np.roll(texture_to_patch, -ye + overlap, axis=0)[:block_size, xs + overlap:xe + overlap]
    adj_lft_blk = np.roll(texture_to_patch, -xs - overlap * 2, axis=1)[ys:ye, -block_size:]
    adj_rgt_blk = np.roll(texture_to_patch, -xs - block_size, axis=1)[ys:ye, :block_size]
    patch = find_patch_vx(adj_lft_blk, adj_rgt_blk, adj_top_blk, adj_btm_blk, lookup_texture, gen_args, rng)
    patch = get_4way_min_cut_patch(adj_lft_blk, adj_rgt_blk, adj_top_blk, adj_btm_blk, patch, gen_args)
    texture_to_patch[ys:ye, xs + overlap:xe + overlap] = patch
    if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(1):
        return None

    return texture_to_patch
