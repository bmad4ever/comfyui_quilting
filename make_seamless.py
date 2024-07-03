import numpy as np
from math import ceil
from itertools import product
# from .quilting.generate import inf, getMinCutPatchBoth
import cv2 as cv  # if possible, remove this dependency

#==================================================


inf = float('inf')


# IMPORTANT NOTES :
# *  find_horizontal_transition_patch uses the maximum of the errors from all adjacency instead of the sum
# This might actually be better even when generating some textures
#   since it minimizes the "worst" combo, not the sum of errors.
# After testing, maybe should consider changing the texture generation to do the same
# find_vertical_transition_patch uses sum and HAS NOT BEEN TESTED
# *  code already "adapted" to use blur in the patches masks.
# Mind that blur can make transition worse, only useful in some cases, so, if added, must be OPTIONAL!

# TODO -> allow to setup different error criteria. e.g. max median, sum of means, etc...
# TODO -> allow to blur patches' edges
#  Ideally these options should also be available in the previous implemented quilting solution,
#  so it might require some refactoring... ( O__o) hmmmm... ... ...

#==================================================
def find_horizontal_transition_patch(refBlockLeft, refBlockRight, refBlockTop, texture, blocksize, overlap, tolerance,
                                     rng: np.random.Generator):
    '''
    Find the best horizontal patch to fit between two blocks.
    Top block is optional; if provided is also accounted for when computing the error.
    '''
    H, W = texture.shape[:2]
    errMat = np.zeros((H - blocksize, W - blocksize)) + inf
    bmo = blocksize - overlap
    for i, j in product(range(H - blocksize), range(W - blocksize)):
        rmsVal = ((texture[i:i + blocksize, j + bmo:j + blocksize] - refBlockRight[:, :overlap]) ** 2).mean()
        rmsVal = np.maximum(rmsVal,
                            ((texture[i:i + blocksize, j:j + overlap] - refBlockLeft[:, -overlap:]) ** 2).mean())
        if refBlockTop is not None:
            rmsVal = np.maximum(rmsVal,
                                ((texture[i:i + overlap, j:j + blocksize] - refBlockTop[-overlap:, :]) ** 2).mean())

        if rmsVal > 0:
            errMat[i, j] = rmsVal

    minVal = np.min(errMat)
    y, x = np.nonzero(errMat < (1.0 + tolerance) * minVal)
    c = rng.integers(len(y))
    y, x = y[c], x[c]
    return texture[y:y + blocksize, x:x + blocksize]


def find_4way_patch(refBlockLeft, refBlockRight, refBlockTop, refBlockBottom, texture, blocksize, overlap, tolerance,
                    rng: np.random.Generator):
    H, W = texture.shape[:2]
    errMat = np.zeros((H - blocksize, W - blocksize)) + inf
    bmo = blocksize - overlap
    for i, j in product(range(H - blocksize), range(W - blocksize)):
        rmsVal = ((texture[i:i + blocksize, j + bmo:j + blocksize] - refBlockRight[:, :overlap]) ** 2).mean()
        rmsVal = np.maximum(rmsVal,
                            ((texture[i:i + blocksize, j:j + overlap] - refBlockLeft[:, -overlap:]) ** 2).mean())
        rmsVal = np.maximum(rmsVal, ((texture[i:i + overlap, j:j + blocksize] - refBlockTop[-overlap:, :]) ** 2).mean())
        rmsVal = np.maximum(rmsVal, (
                (texture[i + bmo:i + blocksize, j:j + blocksize] - refBlockBottom[:overlap, :]) ** 2).mean())

        if rmsVal > 0:
            errMat[i, j] = rmsVal

    minVal = np.min(errMat)
    y, x = np.nonzero(errMat < (1.0 + tolerance) * minVal)
    c = rng.integers(len(y))
    y, x = y[c], x[c]
    return texture[y:y + blocksize, x:x + blocksize]


def make_seamless_horizontally(image, block_size, overlap, tolerance, rng: np.random.Generator, ref_image=None):
    assert overlap * 2 <= block_size, "overlap needs to be less or equal to half of the block size"

    if ref_image is None:
        ref_image = image

    bmo = block_size - overlap

    src_h, src_w = image.shape[:2]
    strip_w = block_size - overlap * 2
    out_h, out_w = src_h, src_w + strip_w
    n_h = int(ceil((out_h - block_size) / bmo))

    texture_map_h = block_size + n_h * bmo
    texture_map = np.zeros((texture_map_h, out_w, image.shape[-1]))
    texture_map[:src_h, :src_w] = image
    texture_map = np.roll(texture_map, -block_size, axis=1)  # roll left edge to the right edge
    for y in range(out_h, texture_map_h):
        texture_map[y, :] = texture_map[out_h - 1, :]

    # patch horizontal boundaries
    x1 = out_w - 2 * block_size + overlap
    x2 = x1 + block_size

    # get 1st patch
    # devnote -> could make it at the middle to parallel by 2x; doesn't seem needed though, might reconsider later...
    refBlockLeft = texture_map[:block_size, x1 - bmo:x2 - bmo]
    refBlockRight = texture_map[:block_size, -block_size:]
    patchBlock = find_horizontal_transition_patch(
        refBlockLeft, refBlockRight, None, ref_image, block_size, overlap, tolerance, rng)
    minCutPatch = getMinCutPatch4ways(refBlockLeft, refBlockRight, None, None,
                                      patchBlock, block_size, overlap)
    texture_map[:block_size, x1:x2] = minCutPatch
    #texture_map[:block_size, x1:x2] *= .5 #  debug, check 1st tile placement

    for y in range(1, n_h + 1):
        blk_1y = y * bmo  # block top corner y
        blk_2y = blk_1y + block_size  # block bottom corner y

        # find adjacent blocks, and the min errors independently
        refBlockLeft = texture_map[blk_1y:blk_2y, x1 - bmo:x2 - bmo]
        refBlockRight = texture_map[blk_1y:blk_2y, -block_size:]
        if (top_block_y_offset := blk_1y - block_size + overlap) < 0:
            refBlockTop = np.zeros((block_size, block_size, image.shape[-1]))
            refBlockTop[block_size + top_block_y_offset:, :] = texture_map[0:-top_block_y_offset, x1:x2]
        else:
            refBlockTop = texture_map[(blk_1y - block_size + overlap):(blk_1y + overlap), x1:x2]

        patchBlock = find_horizontal_transition_patch(
            refBlockLeft, refBlockRight, refBlockTop, ref_image, block_size, overlap, tolerance, rng)
        minCutPatch = getMinCutPatch4ways(refBlockLeft, refBlockRight, refBlockTop, None,
                                          patchBlock, block_size, overlap)

        texture_map[blk_1y:blk_2y, x1:x2] = minCutPatch
        #texture_map[blk_1y:blk_2y, x1:x2] *= .5  # debug, check stripe location

        #coord_jobs_array[1 + job_id] += nW
        #if coord_jobs_array[0] > 0:
        #    return textureMap

    return texture_map[:out_h, :out_w]


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


def getMinCutPatch4ways(refBlockLeft, refBlockRight, refBlockTop, refBlockBottom, patchBlock, block_size, overlap):
    mask_left = getMinCutPatchMaskHorizontal(refBlockLeft, patchBlock, block_size, overlap)  # w/ the left block
    mask_right = getMinCutPatchMaskHorizontal(np.fliplr(refBlockRight), np.fliplr(patchBlock), block_size, overlap)
    mask_right = np.fliplr(mask_right)

    # (optional step) blur masks for a more seamless integration ( sometimes makes transition more noticeable, depends )
    #mask_left = cv.blur(mask_left, (5, 5))
    #mask_right = cv.blur(mask_right, (5, 5))

    masks_list = [mask_left, mask_right]

    if refBlockTop is not None:
        # V , >  counterclockwise rotation
        mask_top = getMinCutPatchMaskHorizontal(np.rot90(refBlockTop), np.rot90(patchBlock), block_size, overlap)
        mask_top = np.rot90(mask_top, 3)
        #mask_top = cv.blur(mask_top, (5, 5))
        masks_list.append(mask_top)

    if refBlockBottom is not None:
        mask_bottom = getMinCutPatchMaskHorizontal(np.fliplr(np.rot90(refBlockBottom)),
                                                   np.fliplr(np.rot90(refBlockBottom)), block_size, overlap)
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
    resBlock = mask_left * np.roll(refBlockLeft, overlap, 1)
    resBlock += mask_right * np.roll(refBlockRight, -overlap, 1)
    if refBlockTop is not None:
        resBlock += mask_top * np.roll(refBlockTop, overlap, 0)
    if refBlockBottom is not None:
        resBlock += mask_bottom * np.roll(refBlockBottom, -overlap, 0)
    resBlock *= mask_mos

    # place patch section
    patch_weight = 1 - masks_max
    resBlock = resBlock + patch_weight * patchBlock

    return resBlock


def make_seamless_vertically(image, block_size, overlap, tolerance, rng: np.random.Generator):
    rotated_solution = make_seamless_horizontally(np.rot90(image, 1), block_size, overlap, tolerance, rng)
    return np.rot90(rotated_solution, -1).copy()


def make_seamless_both(image, block_size, overlap, tolerance, rng: np.random.Generator):
    # image dims  >=  block_size + overlap * 2 , must be true so that
    #  there is some overlap area available for the last patch - "the big square".
    big_block_size = block_size + overlap * 2
    assert image.shape[0] >= big_block_size
    assert image.shape[1] >= big_block_size

    # patch the texture in both directions. the last stripe's endpoints won't loop yet.
    vs = make_seamless_vertically(image, block_size, overlap, tolerance, rng)
    hs_vs = make_seamless_horizontally(vs, block_size, overlap, tolerance, rng, ref_image=image)

    # ___ patch vertical stripe from 2nd operation w/ "the big square" ___
    #   center the area to patch 1st, this will make the rolls in the next step easier
    fs = np.roll(hs_vs, block_size - overlap + (hs_vs.shape[1] + block_size) // 2, axis=1)
    fs = np.roll(fs, fs.shape[0] // 2 + block_size - overlap + block_size // 2, axis=0)

    yc = (fs.shape[0] - block_size) // 2
    xc = (fs.shape[1] - block_size) // 2
    t, b, l, r = yc - overlap, yc + block_size + overlap, xc - overlap, xc + block_size + overlap
    #fs[t:b, l:r, :] *= 0.8  # debug, check roi
    adj_top_blk = np.roll(fs, yc + block_size, axis=0)[-big_block_size:, l:r]
    adj_btm_blk = np.roll(fs, -yc - block_size, axis=0)[:big_block_size, l:r]
    adj_lft_blk = np.roll(fs, xc + block_size, axis=1)[t:b, -big_block_size:]
    adj_rgt_blk = np.roll(fs, -xc - block_size, axis=1)[t:b, :big_block_size]
    patch = find_4way_patch(adj_lft_blk, adj_rgt_blk, adj_top_blk, adj_btm_blk,
                            image, big_block_size, overlap, tolerance, rng)
    patch = getMinCutPatch4ways(adj_lft_blk, adj_rgt_blk, adj_top_blk, adj_btm_blk,
                                patch, big_block_size, overlap)
    fs[t:b, l:r, :] = patch
    #fs[t:b, l:r, :] *= .5 # debug, check last patch bounds
    return fs


if __name__ == "__main__":
    img = cv.imread("./t8.png", cv.IMREAD_COLOR)
    rng = np.random.default_rng(1300)

    #img_sh = make_seamless_horizontally(img, 64, 30, .001, rng)
    #img_sh_tiled = np.empty((img_sh.shape[0], img_sh.shape[1]*2, img_sh.shape[2]))
    #img_sh_tiled[:, :img_sh.shape[1]] = img_sh
    #img_sh_tiled[:, img_sh.shape[1]:] = img_sh
    #cv.imwrite("./h_seamless.png", img_sh_tiled)

    #img_sv = make_seamless_vertically(img, 36, 12, .001, rng)
    #img_sv_tiled = np.empty((img_sv.shape[0]*2, img_sv.shape[1], img_sv.shape[2]))
    #img_sv_tiled[:img_sv.shape[0], :] = img_sv
    #img_sv_tiled[img_sv.shape[0]:, :] = img_sv
    #cv.imwrite("./v_seamless.png", img_sv_tiled)

    #    Dev Note 03/07 -> w/ respect to early tests:
    # stripes & "big square" can be very noticeable if the block_size isn't "just right".
    # I'm not feeling like implementing an alternative approach, such as using
    # two polar unwraps for the inner square, which may not even yield better results.
    # what would be an easy & cheap way to estimate good size?
    # check scale inv. features sizes or distances? maybe check fourier ?? no clear idea yet

    #    Dev Note 03/07 -> w/ respect to potential usage:
    # Unsure if this would be useful in latent space.
    # Likely less noticeable due to lower resolution...
    # but if diffusion is used the edges should change,
    # so unless there is some safeguard in place it should be ratter useless.

    bs = round(min(img.shape[:2]) / 5)
    overlap = round(bs / 2.2)
    print(f"block size = {bs}  ;  overlap = {overlap}")
    img_sb = make_seamless_both(img, bs, overlap, .15, rng)  # .1 seems to low ; perhaps .2 is good
    cv.imwrite("./b_seamless.png", img_sb)

    img_4tiles = np.empty((img_sb.shape[0] * 2, img_sb.shape[1] * 2, img_sb.shape[2]))
    img_4tiles[:img_sb.shape[0], :img_sb.shape[1]] = img_sb
    img_4tiles[img_sb.shape[0]:, :img_sb.shape[1]] = img_sb
    img_4tiles[:img_sb.shape[0], img_sb.shape[1]:] = img_sb
    img_4tiles[img_sb.shape[0]:, img_sb.shape[1]:] = img_sb
    cv.imwrite("./4b_seamless.png", img_4tiles)
