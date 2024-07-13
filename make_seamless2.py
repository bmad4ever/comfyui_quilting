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


def seamless_horizontal(use_parallel, image, block_size, overlap, tolerance, version, ags, rng,
                        uicd: UiCoordData | None = None):
    """
    @param ags: auxiliar generation scale
    """
    out_w = image.shape[1] * ags
    out_h = image.shape[0] * ags

    #___________________________________________________________________________________
    aux_texture = (
        generate_texture_parallel(image, block_size, overlap, out_h, out_w,
                                  tolerance, version, 1, rng, None)
        if use_parallel
        else generate_texture(image, block_size, overlap, out_h, out_w,
                              tolerance, version, rng, None)
    )

    def just_h_extension_BAD_IDEA_DELETE_LATER(out_h, out_w):
        n_h = int(np.ceil((out_h - block_size) / (block_size - overlap)))
        n_w = int(np.ceil((out_w - block_size) / (block_size - overlap)))
        out_w = n_w * (block_size - overlap)
        out_h = block_size + n_h * (block_size - overlap)

        #aux_texture = np.empty((out_h, image.shape[1] + out_w, image.shape[2])).astype(image.dtype)
        aux_texture = np.zeros(
            (out_h,
             image.shape[1] + out_w,
             image.shape[2])).astype(image.dtype)
        aux_texture[:image.shape[0], :image.shape[1]] = image

        # extend bottom pixels
        for i in range(image.shape[0], out_h):
            aux_texture[i, :image.shape[1]] = aux_texture[i - 1, :image.shape[1]]

        start_block = image[:block_size, -block_size:]
        aux_texture[:block_size, image.shape[1] - block_size:] = fill_row(
            image, start_block, overlap, n_w, tolerance, version, rng)

        if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(n_w):
            return None

        aux_texture[:, image.shape[1] - block_size:] = fill_quad(n_h, n_w, block_size, overlap,
                                                                 aux_texture[:, image.shape[1] - block_size:],
                                                                 image, tolerance, version, rng, uicd)

        return aux_texture

    #___________________________________________________________________________________

    image = np.roll(image, +block_size//2, axis=1)

    # left & right overlap errors
    lo_errs = cv.matchTemplate(image=aux_texture[:, :-block_size],
                               templ=image[:overlap, :], method=cv.TM_CCOEFF_NORMED)
    ro_errs = cv.matchTemplate(image=np.roll(aux_texture, -block_size + overlap, axis=1)[:, :-block_size],
                               templ=image[block_size - overlap:block_size, :], method=cv.TM_CCOEFF_NORMED)

    err_mat = np.add(lo_errs, ro_errs) if version <= 1 else np.minimum(lo_errs, ro_errs)
    min_val = np.min(err_mat[err_mat > 0 if tolerance > 0 else True])  # ignore zeroes to enforce tolerance usage
    y, x = np.nonzero(err_mat <= min_val)  # ignore tolerance here, choose best
    c = rng.integers(len(y))
    y, x = y[c], x[c]

    # "fake" block will only contain the overlap, in order to re-use existing function.
    fake_left_block = np.zeros((image.shape[0], image.shape[0], image.shape[2])).astype(image.dtype)
    fake_right_block = np.zeros((image.shape[0], image.shape[0], image.shape[2])).astype(image.dtype)
    fake_left_block[:, -overlap:] = image[:, :overlap]
    fake_right_block[:, :overlap] = image[:, block_size - overlap:block_size]
    fake_block_sized_patch = np.zeros((image.shape[0], image.shape[0], image.shape[2])).astype(image.dtype)
    fake_block_sized_patch[:, :overlap] = aux_texture[y:y + image.shape[0], x:x + overlap]
    fake_block_sized_patch[:, -overlap:] = aux_texture[y:y + image.shape[0], x + block_size - overlap:x + block_size]
    left_side_patch = getMinCutPatchHorizontal(fake_left_block, fake_block_sized_patch, image.shape[0], overlap)
    right_side_patch = np.fliplr(
        getMinCutPatchHorizontal(np.fliplr(fake_right_block), np.fliplr(fake_block_sized_patch), image.shape[0],
                                 overlap))

    # paste vertical stripe patch
    image[:, :block_size] = aux_texture[y:y + image.shape[0], x:x + block_size]
    image[:, :overlap] = left_side_patch[:, :overlap]
    image[:, block_size - overlap:block_size] = right_side_patch[:, -overlap:]
    return image


def seamless_vertical(use_parallel, image, block_size, overlap, tolerance, version, ags, rng,
                      uicd: UiCoordData | None = None):
    rotated_solution = seamless_horizontal(use_parallel, np.rot90(image), block_size,
                                           overlap, tolerance, version, ags, rng, uicd)
    return np.rot90(rotated_solution, -1).copy()

# seamless both needs to patch small seam on last stripe, similar to previous implementation
# just need to check the offsets and adjust in either function,
# so that the seam is positioned the same way in both solutions; then just reuse the squares patches code

if __name__ == "__main__":
    img = cv.imread("./t166.png", cv.IMREAD_COLOR)
    rng = np.random.default_rng(1300)

    bs = round(img.shape[0]/1.6)
    ov = round(.38* bs)

    img_sh = seamless_horizontal(True, img, bs, ov, .1, 1, 50, rng, None)
    img_sh_tiled = np.empty((img_sh.shape[0], img_sh.shape[1] * 2, img_sh.shape[2]))
    img_sh_tiled[:, :img_sh.shape[1]] = img_sh
    img_sh_tiled[:, img_sh.shape[1]:] = img_sh
    cv.imwrite("./h_seamless_alt.png", img_sh_tiled)

    #img_sv = seamless_vertical(True, img, bs, ov, .0001, 1, 16, rng, None)
    #img_sv_tiled = np.empty((img_sv.shape[0] * 2, img_sv.shape[1], img_sv.shape[2]))
    #img_sv_tiled[:img_sv.shape[0], :] = img_sv
    #img_sv_tiled[img_sv.shape[0]:, :] = img_sv
    #cv.imwrite("./v_seamless_alt.png", img_sv_tiled)
