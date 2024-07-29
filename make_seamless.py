import numpy as np
from math import ceil
import cv2 as cv  # if possible, remove this dependency

from custom_nodes.comfyui_quilting.quilting import generate_texture, generate_texture_parallel, fill_row, fill_quad
from custom_nodes.comfyui_quilting.jena2020.generate import getMinCutPatchHorizontal
from custom_nodes.comfyui_quilting.patch_search import get_generic_find_patch_method
from .types import UiCoordData

#==================================================


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
    assert overlap * 2 <= block_size, "overlap needs to be less or equal to half of the block size"
    assert block_size <= image.shape[0], "block size must be smaller or equal to the image's height"
    assert block_size <= image.shape[1], "block size must be smaller or equal to the image's width "

    if lookup_texture is None:
        lookup_texture = image
    else:
        assert lookup_texture.dtype == image.dtype, "lookup_texture dtype does not match image dtype"

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


def getMinCutPatchMaskHorizontal(block1, block2, blocksize, overlap):
    '''
	Get the min cut patch done horizontally ( block1 is to the left of block2
	'''
    err = ((block1[:, -overlap:] - block2[:, :overlap]) ** 2).mean(2)
    # maintain minIndex for 2nd row onwards and
    minIndex = []
    E = [list(err[0])]
    for i in range(1, err.shape[0]):
        # Get min values and args, -1 = left, 0 = middle, 1 = right
        e = [inf] + E[-1] + [inf]
        e = np.array([e[:-2], e[1:-1], e[2:]])
        # Get minIndex
        minArr = e.min(0)
        minArg = e.argmin(0) - 1
        minIndex.append(minArg)
        # Set Eij = e_ij + min_
        Eij = err[i] + minArr
        E.append(list(Eij))

    # Check the last element and backtrack to find path
    path = []
    minArg = np.argmin(E[-1])
    path.append(minArg)

    # Backtrack to min path
    for idx in minIndex[::-1]:
        minArg = minArg + idx[minArg]
        path.append(minArg)
    # Reverse to find full path
    path = path[::-1]
    mask = np.zeros((blocksize, blocksize, block1.shape[2]))
    for i in range(len(path)):
        mask[i, :path[i] + 1] = 1

    #resBlock = np.zeros(block1.shape)
    #resBlock[:, :overlap] = block1[:, -overlap:]
    #resBlock = resBlock*mask + block2*(1-mask)
    # resBlock = block1*mask + block2*(1-mask)
    return mask


def get_4way_min_cut_patch(ref_block_left, ref_block_right, ref_block_top, ref_block_bottom,
                           patch_block, block_size, overlap):
    mask_left = getMinCutPatchMaskHorizontal(ref_block_left, patch_block, block_size, overlap)  # w/ the left block
    # (optional step) blur masks for a more seamless integration ( sometimes makes transition more noticeable, depends )
    #mask_left = cv.blur(mask_left, (5, 5))

    masks_list = [mask_left]

    if ref_block_right is not None:
        mask_right = getMinCutPatchMaskHorizontal(np.fliplr(ref_block_right), np.fliplr(patch_block), block_size,
                                                  overlap)
        mask_right = np.fliplr(mask_right)
        # mask_right = cv.blur(mask_right, (5, 5))
        masks_list.append(mask_right)

    if ref_block_top is not None:
        # V , >  counterclockwise rotation
        mask_top = getMinCutPatchMaskHorizontal(np.rot90(ref_block_top), np.rot90(patch_block), block_size, overlap)
        mask_top = np.rot90(mask_top, 3)
        #mask_top = cv.blur(mask_top, (5, 5))
        masks_list.append(mask_top)

    if ref_block_bottom is not None:
        mask_bottom = getMinCutPatchMaskHorizontal(np.fliplr(np.rot90(ref_block_bottom)),
                                                   np.fliplr(np.rot90(ref_block_bottom)), block_size, overlap)
        mask_bottom = np.rot90(np.fliplr(mask_bottom), 3)
        #mask_bottom = cv.blur(mask_bottom, (5, 5))
        masks_list.append(mask_bottom)

    # --- apply masks and return block ---
    # compute auxiliary data
    mask_s = sum(masks_list)
    masks_max = np.maximum.reduce(masks_list)
    mask_mos = np.divide(masks_max, mask_s, out=np.zeros_like(mask_s), where=mask_s != 0)
    # note -> if blurred, masks weights are scaled with respect the max mask value.
    # example: mask1: 0.2 mask2:0.5 -> max = .5 ; sum = .7 -> (.2*lb + .5*tb) * (max/sum) + patch * (1 - max)

    # place adjacent block sections
    resBlock = mask_left * np.roll(ref_block_left, overlap, 1)
    if ref_block_right is not None:
        resBlock += mask_right * np.roll(ref_block_right, -overlap, 1)
    if ref_block_top is not None:
        resBlock += mask_top * np.roll(ref_block_top, overlap, 0)
    if ref_block_bottom is not None:
        resBlock += mask_bottom * np.roll(ref_block_bottom, -overlap, 0)
    resBlock *= mask_mos

    # place patch section
    patch_weight = 1 - masks_max
    resBlock = resBlock + patch_weight * patch_block

    return resBlock


def make_seamless_vertically(image, block_size, overlap, tolerance, rng: np.random.Generator,
                             version: int = 1, lookup_texture=None, uicd: UiCoordData | None = None):
    rotated_solution = make_seamless_horizontally(
        np.rot90(image, 1), block_size, overlap, tolerance, rng=rng, version=version,
        lookup_texture=None if lookup_texture is None else np.rot90(lookup_texture, 1),
        uicd=uicd)
    return np.rot90(rotated_solution, -1).copy() if rotated_solution is not None else None


def make_seamless_both(image, block_size, overlap, tolerance, rng: np.random.Generator,
                       version: int = 1, lookup_texture=None, uicd: UiCoordData | None = None):
    assert image.shape[0] >= block_size
    assert image.shape[1] >= block_size + overlap * 2

    lookup_texture = image if lookup_texture is None else lookup_texture
    assert lookup_texture.shape[0] >= block_size
    assert lookup_texture.shape[1] >= block_size + overlap * 2

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

    return patch_horizontal_seam(texture, lookup_texture, block_size, overlap, tolerance, rng, version=version, uicd=uicd)


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


if __name__ == "__main__":
    img = cv.imread("./t166.png", cv.IMREAD_COLOR)
    rng = np.random.default_rng(1300)

    ags = 30
    use_parallel = True
    bs = round(img.shape[0] / 2.6)
    #bs = 168
    ov = round(.4 * bs)
    print(f"block size = {bs}  ;  overlap = {ov}")

    tolerance = 0
    version = 1
    out_w = img.shape[1] * ags
    out_h = img.shape[0] * ags


    def gen_lookup():
        lookup_texture = (
            generate_texture_parallel(img, bs, ov, out_h, out_w,
                                      tolerance, version, 1, rng, None)
            if use_parallel
            else generate_texture(img, bs, ov, out_h, out_w,
                                  tolerance, version, rng, None)
        )
        cv.imwrite("./lookup.png", lookup_texture)


    #gen_lookup()
    #quit()

    lookup_texture = cv.imread("./lookup.png", cv.IMREAD_COLOR)

    #img_sh = make_seamless_horizontally(img, bs, ov, 0, rng, lookup_texture=lookup_texture)
    #img_sh_tiled = np.empty((img_sh.shape[0], img_sh.shape[1]*2, img_sh.shape[2]))
    #img_sh_tiled[:, :img_sh.shape[1]] = img_sh
    #img_sh_tiled[:, img_sh.shape[1]:] = img_sh
    #cv.imwrite("./h_seamless.png", img_sh_tiled)
    #quit()

    #img_sv = make_seamless_vertically(img, 64, 30, .00001, rng, lookup_texture=lookup_texture)
    #img_sv_tiled = np.empty((img_sv.shape[0]*2, img_sv.shape[1], img_sv.shape[2]))
    #img_sv_tiled[:img_sv.shape[0], :] = img_sv
    #img_sv_tiled[img_sv.shape[0]:, :] = img_sv
    #cv.imwrite("./v_seamless.png", img_sv_tiled)

    #    Dev Note 03/07 -> w/ respect to potential usage:
    # Unsure if this would be useful in latent space.
    # Likely less noticeable due to lower resolution...
    # but if diffusion is used the edges should change,
    # so unless there is some safeguard in place it should be ratter useless.

    #bs = 73  #round(min(img.shape[:2]) / 5)
    #overlap = round(bs / 2.3)
    img_sb = make_seamless_both(img, bs, ov, 0, rng, version=1,
                                lookup_texture=lookup_texture)  # .1 seems to low ; perhaps .2 is good
    cv.imwrite("./b_seamless_v2.png", img_sb)

    img_4tiles = np.empty((img_sb.shape[0] * 2, img_sb.shape[1] * 2, img_sb.shape[2]))
    img_4tiles[:img_sb.shape[0], :img_sb.shape[1]] = img_sb
    img_4tiles[img_sb.shape[0]:, :img_sb.shape[1]] = img_sb
    img_4tiles[:img_sb.shape[0], img_sb.shape[1]:] = img_sb
    img_4tiles[img_sb.shape[0]:, img_sb.shape[1]:] = img_sb
    cv.imwrite("./4b_seamless_v2.png", img_4tiles)
