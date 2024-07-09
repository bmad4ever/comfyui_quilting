import numpy as np
import cv2 as cv

epsilon = np.finfo(float).eps


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

    err_mat = np.maximum.reduce(blks_sqdiffs)
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
    This version makes use of TM_CCOEFF in matchTemplate instead of TM_SQDIFF.
    """
    blks_ccs = []
    t_max = 0
    if ref_block_left is not None:
        blk_overlap = ref_block_left[:, -overlap:]
        blks_ccs.append(cv.matchTemplate(
            image=texture[:, :-block_size + overlap],
            templ=blk_overlap, method=cv.TM_CCOEFF))
        t_max = cv.matchTemplate(blk_overlap, blk_overlap, method=cv.TM_CCOEFF)[0]
    if ref_block_right is not None:
        blk_overlap = ref_block_right[:, :overlap]
        blks_ccs.append(cv.matchTemplate(
            image=np.roll(texture, -block_size + overlap, axis=1)[:, :-block_size + overlap],
            templ=blk_overlap, method=cv.TM_CCOEFF))
        t_max += cv.matchTemplate(blk_overlap, blk_overlap, method=cv.TM_CCOEFF)[0]
    if ref_block_top is not None:
        blk_overlap = ref_block_top[-overlap:, :]
        blks_ccs.append(cv.matchTemplate(
            image=texture[:-block_size + overlap, :],
            templ=blk_overlap, method=cv.TM_CCOEFF))
        t_max += cv.matchTemplate(blk_overlap, blk_overlap, method=cv.TM_CCOEFF)[0]
    if ref_block_bottom is not None:
        blk_overlap = ref_block_bottom[:overlap, :]
        blks_ccs.append(cv.matchTemplate(
            image=np.roll(texture, -block_size + overlap, axis=0)[:-block_size + overlap, :],
            templ=blk_overlap, method=cv.TM_CCOEFF))
        t_max += cv.matchTemplate(blk_overlap, blk_overlap, method=cv.TM_CCOEFF)[0]

    err_mat = t_max - np.add.reduce(blks_ccs)
    print(f"min val = {np.min(err_mat)}")
    min_val = np.min(err_mat[err_mat > 0 if tolerance > 0 else True])  # ignore zeroes to enforce tolerance usage
    y, x = np.nonzero(err_mat <= (1.0 + tolerance) * min_val)
    c = rng.integers(len(y))
    y, x = y[c], x[c]
    return texture[y:y + block_size, x:x + block_size]
