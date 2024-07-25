# An alternative approach to making the texture seamless
import numpy as np
import cv2 as cv
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from custom_nodes.comfyui_quilting.types import UiCoordData

from custom_nodes.comfyui_quilting.quilting import generate_texture, generate_texture_parallel, fill_row, fill_quad
from custom_nodes.comfyui_quilting.make_seamless import get_4way_min_cut_patch
from custom_nodes.comfyui_quilting.jena2020.generate import getMinCutPatchHorizontal
from make_seamless import find_4way_patch_v2, patch_horizontal_seam


def seamless_horizontal( image, block_size, overlap, version, lookup_texture, rng,
                        uicd: UiCoordData | None = None):
    """
    @param ags: auxiliar generation scale
    """
    image = np.roll(image, +block_size//2, axis=1)

    # left & right overlap errors
    lo_errs = cv.matchTemplate(image=lookup_texture[:, :-block_size],
                               templ=image[:, :overlap], method=cv.TM_SQDIFF)
    ro_errs = cv.matchTemplate(image=np.roll(lookup_texture, -block_size + overlap, axis=1)[:, :-block_size],
                               templ=image[:, block_size - overlap:block_size], method=cv.TM_SQDIFF)

    err_mat = np.add(lo_errs, ro_errs) if version <= 1 else np.minimum(lo_errs, ro_errs)
    min_val = np.min(err_mat) #err_mat[err_mat > 0 if tolerance > 0 else True])  # ignore zeroes to enforce tolerance usage
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
    left_side_patch = getMinCutPatchHorizontal(fake_left_block, fake_block_sized_patch, image.shape[0], overlap)
    right_side_patch = np.fliplr(
        getMinCutPatchHorizontal(np.fliplr(fake_right_block), np.fliplr(fake_block_sized_patch), image.shape[0],
                                 overlap))

    # paste vertical stripe patch
    image[:, :block_size] = lookup_texture[y:y + image.shape[0], x:x + block_size]
    image[:, :overlap] = left_side_patch[:, :overlap]
    image[:, block_size - overlap:block_size] = right_side_patch[:, -overlap:]
    return image


def seamless_vertical(image, block_size, overlap, version, lookup_texture, rng,
                      uicd: UiCoordData | None = None):
    rotated_solution = seamless_horizontal(np.rot90(image), block_size, overlap,
                                           version, np.rot90(lookup_texture), rng, uicd)
    return np.rot90(rotated_solution, -1).copy()

# seamless both needs to patch small seam on last stripe, similar to previous implementation
# just need to check the offsets and adjust in either function,
# so that the seam is positioned the same way in both solutions; then just reuse the squares patches code

def seamless_both(image, block_size, overlap, version, lookup_texture, rng,
                      uicd: UiCoordData | None = None):
    assert image.shape[0] >= block_size
    assert image.shape[1] >= block_size + overlap * 2

    texture = seamless_vertical(image, block_size, overlap, version, lookup_texture, rng, uicd)
    texture = seamless_horizontal(texture, block_size, overlap, version, lookup_texture, rng, uicd)

    #texture[:block_size//4, :block_size//2] = 0  # debug only
    #texture[-block_size//4:,:block_size//2] = 0  # debug only

    # center seam & patch it
    texture = np.roll(texture, texture.shape[0]//2, axis=0)
    texture = np.roll(texture, texture.shape[1]//2 - block_size//2, axis=1)
    texture = patch_horizontal_seam(texture, lookup_texture, block_size, overlap, rng)

    return texture


if __name__ == "__main__":
    img = cv.imread("./t166.png", cv.IMREAD_COLOR)
    print(img.dtype)
    rng = np.random.default_rng(1300)

    ags = 30
    use_parallel = True
    bs = round(img.shape[0]/2.25)
    ov = round(.38 * bs)
    tolerance = .0001
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




    #img_sh = seamless_horizontal(img, bs, ov, 1, lookup_texture,rng, None)
    #img_sh_tiled = np.empty((img_sh.shape[0], img_sh.shape[1] * 2, img_sh.shape[2]))
    #img_sh_tiled[:, :img_sh.shape[1]] = img_sh
    #img_sh_tiled[:, img_sh.shape[1]:] = img_sh
    #cv.imwrite("./h_seamless_alt.png", img_sh_tiled)
    #quit()

    #img_sv = seamless_vertical(img, bs, ov, 1, lookup_texture,rng, None)
    #img_sv_tiled = np.empty((img_sv.shape[0] * 2, img_sv.shape[1], img_sv.shape[2]))
    #img_sv_tiled[:img_sv.shape[0], :] = img_sv
    #img_sv_tiled[img_sv.shape[0]:, :] = img_sv
    #cv.imwrite("./v_seamless_alt.png", img_sv_tiled)

    img_sb = seamless_both(img, bs, ov, 1, lookup_texture, rng, None)
    img_sb_tiled = np.empty((img_sb.shape[0] * 2, img_sb.shape[1] * 2, img_sb.shape[2]))
    img_sb_tiled[:img_sb.shape[0], :img_sb.shape[1]] = img_sb
    img_sb_tiled[img_sb.shape[0]:, :img_sb.shape[1]] = img_sb
    img_sb_tiled[:img_sb.shape[0], img_sb.shape[1]:] = img_sb
    img_sb_tiled[img_sb.shape[0]:, img_sb.shape[1]:] = img_sb
    cv.imwrite("./b_seamless_alt.png", img_sb_tiled)
