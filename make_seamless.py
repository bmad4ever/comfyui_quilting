import numpy as np
from math import ceil
from itertools import product
# from .quilting.generate import inf, getMinCutPatchBoth
import cv2 as cv   # if possible, remove this dependency

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
def find_horizontal_transition_patch(refBlockLeft, refBlockRigth, refBlockTop, texture, blocksize, overlap, tolerance,
                                     rng: np.random.Generator):
    '''
    Find the best horizontal patch to fit between two blocks.
    Top block is optional; if provided is also accounted for when computing the error.
    '''
    H, W = texture.shape[:2]
    errMat = np.zeros((H - blocksize, W - blocksize)) + inf
    for i, j in product(range(H - blocksize), range(W - blocksize)):
        rmsVal = ((texture[i:i + blocksize, j + blocksize - overlap:j + blocksize] - refBlockRigth[:,
                                                                                     :overlap]) ** 2).mean()
        rmsVal = np.maximum(rmsVal, ((texture[i:i + blocksize, j:j + overlap] - refBlockLeft[:, -overlap:]) ** 2).mean())
        if refBlockTop is not None:
            rmsVal = np.maximum(rmsVal, ((texture[i:i + overlap, j:j + blocksize] - refBlockTop[-overlap:, :]) ** 2).mean())

        if rmsVal > 0:
            errMat[i, j] = rmsVal

    minVal = np.min(errMat)
    y, x = np.nonzero(errMat < (1.0 + tolerance) * (minVal))
    c = rng.integers(len(y))
    y, x = y[c], x[c]
    return texture[y:y + blocksize, x:x + blocksize]


def find_vertical_transition_patch(refBlockBottom, refBlockTop, texture, blocksize, overlap, tolerance,
                                   rng: np.random.Generator):
    '''
    Find the best vertical patch to fit between two blocks
    Left block is optional; if provided is also accounted for when computing the error.
    '''
    H, W = texture.shape[:2]
    errMat = np.zeros((H - blocksize, W - blocksize)) + inf
    for i, j in product(range(H - blocksize), range(W - blocksize)):
        rmsVal = ((texture[i:i + overlap, j:j + blocksize] - refBlockTop[-overlap:, :]) ** 2).mean()
        rmsVal = rmsVal + ((texture[i + blocksize - overlap:i + blocksize, j:j + blocksize] - refBlockBottom[:overlap,
                                                                                              :]) ** 2).mean()
        if rmsVal > 0:
            errMat[i, j] = rmsVal

    minVal = np.min(errMat)
    y, x = np.nonzero(errMat < (1.0 + tolerance) * (minVal))
    c = rng.integers(len(y))
    y, x = y[c], x[c]
    return texture[y:y + blocksize, x:x + blocksize]


def make_seamless_horizontally(image, block_size, overlap, tolerance, rng: np.random.Generator):
    assert overlap * 2 <= block_size, "overlap needs to be less or equal to half of the block size"

    bmo = block_size - overlap

    src_h, src_w = image.shape[:2]
    strip_w = block_size - overlap * 2
    out_h, out_w = src_h, src_w + strip_w
    n_h = int(ceil((out_h - block_size) / bmo))

    texture_map_h = block_size + n_h*bmo
    texture_map = np.zeros((texture_map_h, out_w, image.shape[-1]))
    texture_map[:src_h, :src_w] = image
    texture_map = np.roll(texture_map, -block_size, axis=1)  # roll left edge to the right edge
    for y in range(out_h, texture_map_h):
        texture_map[y, :] = texture_map[out_h-1, :]

    # patch horizontal boundaries
    x1 = out_w - 2 * block_size + overlap
    x2 = x1 + block_size

    # get 1st patch
    # devnote -> could make it at the middle to parallel by 2x; doesn't seem needed though, might reconsider later...
    refBlockLeft = texture_map[:block_size, x1 - bmo:x2 - bmo]
    refBlockRight = texture_map[:block_size, -block_size:]
    patchBlock = find_horizontal_transition_patch(
        refBlockLeft, refBlockRight, None, image, block_size, overlap, tolerance, rng)
    minCutPatch = getMinCutPatchTri(refBlockLeft, refBlockRight, None, patchBlock, block_size, overlap)
    texture_map[:block_size, x1:x2] = patchBlock #minCutPatch

    for y in range(1, n_h+1):
        blk_1y = y * bmo                # block top corner y
        blk_2y = blk_1y + block_size    # block bottom corner y

        # find adjacent blocks, and the min errors independently
        refBlockLeft = texture_map[blk_1y:blk_2y, x1-bmo:x2-bmo]
        refBlockRight = texture_map[blk_1y:blk_2y, -block_size:]
        if (top_block_y_offset := blk_1y - block_size + overlap) < 0:
            refBlockTop = np.zeros((block_size, block_size, image.shape[-1]))
            refBlockTop[block_size+top_block_y_offset:, :] = texture_map[0:-top_block_y_offset, x1:x2]
        else:
            refBlockTop = texture_map[(blk_1y - block_size + overlap):(blk_1y + overlap), x1:x2]

        patchBlock = find_horizontal_transition_patch(
            refBlockLeft, refBlockRight, refBlockTop, image, block_size, overlap, tolerance, rng)
        minCutPatch = getMinCutPatchTri(refBlockLeft, refBlockRight, refBlockTop, patchBlock, block_size, overlap)

        texture_map[blk_1y:blk_2y, x1:x2] = minCutPatch

        #coord_jobs_array[1 + job_id] += nW
        #if coord_jobs_array[0] > 0:
        #    return textureMap

    # TODO? roll in reverse? not really needed but might be visually clearer for the user to spot the changes.
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


def getMinCutPatchTri(refBlockLeft, refBlockRight, refBlockTop, patchBlock, block_size, overlap):
    mask_left = getMinCutPatchMaskHorizontal(refBlockLeft, patchBlock, block_size, overlap)  # w/ the left block
    mask_right = getMinCutPatchMaskHorizontal(np.fliplr(refBlockRight), np.fliplr(patchBlock), block_size, overlap)
    mask_right = np.fliplr(mask_right)


    # (optional step) blur masks for a more seamless integration ( sometimes makes transition more noticeable, depends )
    #mask_left = cv.blur(mask_left, (5, 5))
    #mask_right = cv.blur(mask_right, (5, 5))

    masks_list = [mask_left, mask_right]

    if refBlockTop is not None:
        mask_top = getMinCutPatchMaskHorizontal(np.rot90(refBlockTop), np.rot90(patchBlock), block_size, overlap)
        mask_top = np.rot90(mask_top, 3)
        #mask_top = cv.blur(mask_top, (5, 5))
        masks_list.append(mask_top)

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
    resBlock *= mask_mos

    # place patch section
    patch_weight = 1 - masks_max
    resBlock = resBlock + patch_weight * patchBlock

    return resBlock


if __name__ == "__main__":
    img = cv.imread("./t9.png", cv.IMREAD_COLOR)
    rng = np.random.default_rng(100)
    img = make_seamless_horizontally(img, 36, 12, .001, rng)
    cv.imwrite("./h_seamless.png", img)
