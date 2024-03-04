from multiprocessing.shared_memory import SharedMemory
from .quilting.generate import *
import cv2 as cv


# -- NOTE -- ___________________________________________________________________________________________________________
#
#   This implementation improves vanilla (a.k.a. sequential) image quilting speed via parallelization.
#
#   The following is a reference to the source implementation contained in the quilting folder:
#       https://github.com/rohitrango/Image-Quilting-for-Texture-Synthesis
#
#   And the original algorithm paper:
#       https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/papers/efros-siggraph01.pdf
#
#   Conceptually, parallelization is done by creating a horizontal and vertical stripes in the shape of a cross
#   dividing the image into 4 sections that can be generated independently.
#   Additionally, each section's rows can also be "generated in parallel" ( clarifications below in 2.2 ).
#
#   The operations are as follows:
#   1.
#       2 stripes for each direction are generated, all in parallel (if enough cores available).
#       1 vertical and 1 horizontal stripes are created with inverted images, in order to re-use the original
#       implementation methods without any additional modifications.
#
#   2.1
#       The 4 separate sections are generated at the same time, using the required inversions,
#       and then flipped at the end so that all can be stitched together
#
#   2.2
#   An OPTIONAL parallelization step is implemented, but improvement is not significant.
#   ( also, can be worse in small generations, with a low number of patches or small source image )
#   While generating each section, instead of generating each row sequentially, multiple rows "can run in parallel" in
#   alternating fashion. The modulo of the row number is run by the job with that number, e.g.:  if using two jobs,
#   one runs the odd rows and another the even rows.
#   Note that one row section CAN ONLY be computed ONCE THE ADJACENT SECTION IN THE PRIOR ROW HAS BEEN COMPUTED.
#   The algorithm is still sequential in nature, this is not some "modern" variant of the algorithm,
#   but it does not require a row to be completely filled in order to compute some of the next row's patches.
#
#   3.
#   Stitching is straightforward, and only needs to take into account the removal
#   of the shared components belonging to the initially generated stripes.
#

# region methods extracted from the source of the referenced implementation for re-usability

def fill_column(image, initial_block, overlap, nH, tolerance, rng: np.random.Generator):
    blocksize = initial_block.shape[0]
    textureMap = np.zeros(
        ((blocksize + nH * (blocksize - overlap)), blocksize, image.shape[2]))
    textureMap[:blocksize, :blocksize, :] = initial_block
    for i, blkIdx in enumerate(range((blocksize - overlap), textureMap.shape[0] - overlap, (blocksize - overlap))):
        refBlock = textureMap[(blkIdx - blocksize + overlap):(blkIdx + overlap), :blocksize]
        patchBlock = findPatchVertical(refBlock, image, blocksize, overlap, tolerance, rng)
        minCutPatch = getMinCutPatchVertical(refBlock, patchBlock, blocksize, overlap)
        textureMap[blkIdx:(blkIdx + blocksize), :blocksize] = minCutPatch
    return textureMap


def fill_row(image, initial_block, overlap, nW, tolerance, rng: np.random.Generator):
    blocksize = initial_block.shape[0]
    textureMap = np.zeros(
        (blocksize, (blocksize + nW * (blocksize - overlap)), image.shape[2]))
    textureMap[:blocksize, :blocksize, :] = initial_block
    for i, blkIdx in enumerate(range((blocksize - overlap), textureMap.shape[1] - overlap, (blocksize - overlap))):
        refBlock = textureMap[:blocksize, (blkIdx - blocksize + overlap):(blkIdx + overlap)]
        patchBlock = findPatchHorizontal(refBlock, image, blocksize, overlap, tolerance, rng)
        minCutPatch = getMinCutPatchHorizontal(refBlock, patchBlock, blocksize, overlap)
        textureMap[:blocksize, (blkIdx):(blkIdx + blocksize)] = minCutPatch
    return textureMap


def fill_quad(nH, nW, blocksize, overlap, textureMap, image, tolerance, rng: np.random.Generator, jobs_shm_name, job_id):
    # here job id is more granular, not the same as generateTextureMap_p, but rather within the sub job
    shm_jobs = SharedMemory(name=jobs_shm_name)
    coord_jobs_array = np.ndarray((2 + job_id,), dtype=np.dtype('uint32'), buffer=shm_jobs.buf)

    for i in range(1, nH + 1):
        for j in range(1, nW + 1):
            blkIndexI = i * (blocksize - overlap)
            blkIndexJ = j * (blocksize - overlap)
            refBlockLeft = textureMap[(blkIndexI):(blkIndexI + blocksize),
                           (blkIndexJ - blocksize + overlap):(blkIndexJ + overlap)]
            refBlockTop = textureMap[(blkIndexI - blocksize + overlap):(blkIndexI + overlap),
                          (blkIndexJ):(blkIndexJ + blocksize)]

            patchBlock = findPatchBoth(refBlockLeft, refBlockTop, image, blocksize, overlap, tolerance, rng)
            minCutPatch = getMinCutPatchBoth(refBlockLeft, refBlockTop, patchBlock, blocksize, overlap)

            textureMap[(blkIndexI):(blkIndexI + blocksize), (blkIndexJ):(blkIndexJ + blocksize)] = minCutPatch

        coord_jobs_array[1+job_id] += nW
        if coord_jobs_array[0] > 0:
            break
        #print("{} out of {} rows complete...".format(i + 1, nH + 1))
    return textureMap


# endregion


def generateTextureMap_p(image, blocksize, overlap, outH, outW, tolerance, nps, rng: np.random.Generator, jobs_shm_name, job_id):
    """
    @param nps: number of parallel stripes; tells how many jobs to use for each of the 4 sections.
    """
    from joblib import Parallel, delayed

    shm_jobs = SharedMemory(name=jobs_shm_name)
    coord_jobs_array = np.ndarray((1 + (1+job_id)*4*nps,), dtype=np.dtype('uint32'), buffer=shm_jobs.buf)
    job_id = job_id * 4 * nps  # offset job id for sub jobs

    # Starting index and block
    H, W = image.shape[:2]
    randH = rng.integers(H - blocksize)
    randW = rng.integers(W - blocksize)

    startBlock = image[randH:randH + blocksize, randW:randW + blocksize]

    # horizontal inverted
    hi_startBlock = cv.flip(startBlock, 1)
    hi_image = cv.flip(image, 1)  # in retrospective, should have used np; but other nodes use cv, so makes no diff.

    # vertical inverted
    vi_startBlock = cv.flip(startBlock, 0)
    vi_image = cv.flip(image, 0)

    # note: inverted both ways is computed later when filling the top-left quadrant/section

    # auxiliary variables
    quad_row_width = ceil(outW / 2 + blocksize / 2)
    quad_column_height = ceil(outH / 2 + blocksize / 2)
    qW = int(ceil((quad_row_width - blocksize) * 1.0 / (blocksize - overlap)))
    qH = int(ceil((quad_column_height - blocksize) * 1.0 / (blocksize - overlap)))
    # might get 1 more row or column than needed here

    # generate 2 vertical strips and 2 horizontal strips that will split the generated canvas in half
    # the center, where the stripes connect, shares the same tile
    args = [
        (image, startBlock, overlap, qW, tolerance, rng),
        (hi_image, hi_startBlock, overlap, qW, tolerance, rng),
        (image, startBlock, overlap, qH, tolerance, rng),
        (vi_image, vi_startBlock, overlap, qH, tolerance, rng)
    ]
    funcs = [fill_row, fill_row, fill_column, fill_column]
    stripes = Parallel(n_jobs=4, backend="loky", timeout=None)(
        delayed(funcs[i])(*args[i]) for i in range(4))
    hs, his, vs, vis = stripes
    coord_jobs_array[1+job_id] = qH*2 + qW*2

    # generate the 4 sections (quadrants)
    args = [
        (vis, his, hi_image, qH, qW, overlap, tolerance, nps, rng, jobs_shm_name, job_id),
        (vis, hs, vi_image, qH, qW, overlap, tolerance, nps, rng, jobs_shm_name, job_id),
        (vs, hs, image, qH, qW, overlap, tolerance, nps, rng, jobs_shm_name, job_id),
        (vs, his, hi_image, qH, qW, overlap, tolerance, nps, rng, jobs_shm_name, job_id)
    ]
    funcs = [quad1, quad2, quad3, quad4]
    quads = Parallel(n_jobs=4, backend="loky", timeout=None)(
        delayed(funcs[i])(*args[i]) for i in range(4))
    q1, q2, q3, q4 = quads

    texture = np.zeros((q1.shape[0] * 2 - blocksize, q1.shape[1] * 2 - blocksize, image.shape[2]))
    bmo = blocksize - overlap
    texture[:q1.shape[0] - bmo, :q1.shape[1] - bmo] = q1[:q1.shape[0] - bmo, :q1.shape[1] - bmo]
    texture[:q1.shape[0] - bmo, q1.shape[1] - bmo:] = q2[:q1.shape[0] - bmo, overlap:]
    texture[q1.shape[0] - bmo:, :q1.shape[1] - bmo] = q4[overlap:, :q1.shape[1] - bmo]
    texture[q1.shape[0] - bmo:, q1.shape[1] - bmo:] = q3[overlap:, overlap:]

    return texture[:outH, :outW]


def quad1(vis, his, hi_image, nH, nW, overlap, tolerance, p_strips, rng, jobs_shm_name, job_id):
    """
    :param his: horizontal inverted stripe
    :param vis: vertical inverted stripe
    """
    vi_hi_s = cv.flip(his, 0)  # vertical inversion of the horizontal inverted stripe
    hi_vi_s = cv.flip(vis, 1)
    vhi_image = cv.flip(hi_image, 0)

    if p_strips > 1:
        size = vis.shape[0] * his.shape[1] * hi_image.shape[2] * hi_image.dtype.itemsize
        shm_text = SharedMemory(create=True, size=size)
        texture = np.ndarray((vis.shape[0], his.shape[1], hi_image.shape[2]), dtype=hi_image.dtype, buffer=shm_text.buf)
    else:
        texture = np.zeros((vis.shape[0], his.shape[1], hi_image.shape[2]))

    texture[:vi_hi_s.shape[0], :vi_hi_s.shape[1]] = vi_hi_s[:, :]
    texture[vi_hi_s.shape[0]:hi_vi_s.shape[0], :hi_vi_s.shape[1]] = hi_vi_s[vi_hi_s.shape[0]:, :]
    if p_strips > 1:
        fill_quad_ps(nH, nW, vi_hi_s.shape[0], overlap, shm_text.name, vhi_image, tolerance, p_strips, rng,
                     jobs_shm_name, job_id+p_strips*0)
    else:
        texture = fill_quad(nH, nW, vi_hi_s.shape[0], overlap, texture, vhi_image, tolerance, rng, jobs_shm_name, job_id+0)
    texture = cv.flip(texture, -1)

    if p_strips > 1:
        shm_text.close()
        shm_text.unlink()
    return texture


def quad2(vis, hs, vi_image, nH, nW, overlap, tolerance, p_strips, rng, jobs_shm_name, job_id):
    vi_hs = cv.flip(hs, 0)

    if p_strips > 1:
        size = vis.shape[0] * hs.shape[1] * vi_image.shape[2] * vi_image.dtype.itemsize
        shm_text = SharedMemory(create=True, size=size)
        texture = np.ndarray((vis.shape[0], hs.shape[1], vi_image.shape[2]), dtype=vi_image.dtype, buffer=shm_text.buf)
    else:
        texture = np.zeros((vis.shape[0], hs.shape[1], vi_image.shape[2]))

    texture[:hs.shape[0], :hs.shape[1]] = vi_hs[:, :]
    texture[hs.shape[0]:vis.shape[0], :vis.shape[1]] = vis[hs.shape[0]:, :]
    if p_strips > 1:
        fill_quad_ps(nH, nW, hs.shape[0], overlap, shm_text.name, vi_image, tolerance, p_strips, rng,
                     jobs_shm_name, job_id+p_strips*1)
    else:
        texture = fill_quad(nH, nW, hs.shape[0], overlap, texture, vi_image, tolerance, rng, jobs_shm_name, job_id+1)
    texture = cv.flip(texture, 0)

    if p_strips > 1:
        shm_text.close()
        shm_text.unlink()
    return texture


def quad4(vs, his, hi_image, nH, nW, overlap, tolerance, p_strips, rng, jobs_shm_name, job_id):
    hi_vs = cv.flip(vs, 1)

    if p_strips > 1:
        size = vs.shape[0] * his.shape[1] * hi_image.shape[2] * hi_image.dtype.itemsize
        shm_text = SharedMemory(create=True, size=size)
        texture = np.ndarray((vs.shape[0], his.shape[1], hi_image.shape[2]), dtype=hi_image.dtype, buffer=shm_text.buf)
    else:
        texture = np.zeros((vs.shape[0], his.shape[1], hi_image.shape[2]))

    texture[:his.shape[0], :his.shape[1]] = his[:, :]
    texture[his.shape[0]:vs.shape[0], :vs.shape[1]] = hi_vs[his.shape[0]:, :]
    if p_strips > 1:
        fill_quad_ps(nH, nW, his.shape[0], overlap, shm_text.name, hi_image, tolerance, p_strips, rng,
                     jobs_shm_name, job_id+p_strips*2)
    else:
        texture = fill_quad(nH, nW, his.shape[0], overlap, texture, hi_image, tolerance, rng, jobs_shm_name, job_id+2)
    texture = cv.flip(texture, 1)

    if p_strips > 1:
        shm_text.close()
        shm_text.unlink()
    return texture


def quad3(vs, hs, image, nH, nW, overlap, tolerance, p_strips, rng, jobs_shm_name, job_id):
    if p_strips > 1:
        size = vs.shape[0] * hs.shape[1] * image.shape[2] * image.dtype.itemsize
        shm_text = SharedMemory(create=True, size=size)
        texture = np.ndarray((vs.shape[0], hs.shape[1], image.shape[2]), dtype=image.dtype, buffer=shm_text.buf)
    else:
        texture = np.zeros((vs.shape[0], hs.shape[1], image.shape[2]))

    texture[:hs.shape[0], :hs.shape[1]] = hs[:, :]
    texture[hs.shape[0]:vs.shape[0], :vs.shape[1]] = vs[hs.shape[0]:, :]
    if p_strips > 1:
        fill_quad_ps(nH, nW, vs.shape[1], overlap, shm_text.name, image, tolerance, p_strips, rng,
                     jobs_shm_name, job_id+p_strips*3)
    else:
        return fill_quad(nH, nW, vs.shape[1], overlap, texture, image, tolerance, rng, jobs_shm_name, job_id+3)
    texture = texture.copy()

    if p_strips > 1:
        shm_text.close()
        shm_text.unlink()
    return texture


def fill_quad_ps(nH, nW, blocksize, overlap, texture_shared_mem_name, image, tolerance, total_procs, rng, jobs_shm_name, job_id):
    from joblib import Parallel, delayed

    bmo = blocksize - overlap
    b_o = ceil(blocksize / (blocksize - overlap))

    # note : ShareableList seems to have problems even though there are no concurrent writes to the same position...

    # setup shared array to store & check each job row & column
    size = 2 * total_procs * np.int32(1).itemsize
    shm_coord = SharedMemory(create=True, size=size)
    np_coord = np.ndarray((2 * total_procs,), dtype=np.int32, buffer=shm_coord.buf)
    for ip in range(total_procs):
        np_coord[2 * ip] = 1 + ip
        np_coord[2 * ip + 1] = 1

    def fill_rows(pid, coord_shared_list_name, texture_shm_name, sub_job_id):
        shm_jobs = SharedMemory(name=jobs_shm_name)
        coord_jobs_array = np.ndarray((2 + sub_job_id,), dtype=np.dtype('uint32'), buffer=shm_jobs.buf)

        shm_coord_ref = SharedMemory(name=coord_shared_list_name)
        coord_list = np.ndarray((2 * total_procs,), dtype=np.int32, buffer=shm_coord_ref.buf)
        prior_proc_base_index = (pid + total_procs - 1) % total_procs
        shm_texture = SharedMemory(name=texture_shm_name)
        texture = np.ndarray((blocksize + nH * bmo, blocksize + nW * bmo, image.shape[2]), dtype=image.dtype,
                             buffer=shm_texture.buf)

        for i in range(1 + pid, nH + 1, total_procs):
            coord_list[pid * 2 + 0] = i
            for j in range(1, nW + 1):
                coord_list[pid * 2 + 1] = j

                # if previous row hasn't processed the adjacent section yet wait for it to advance.
                # -1 is used to shortcircuit this check when the job on the prior row has completed all rows.
                while -1 < coord_list[prior_proc_base_index * 2 + 0] < i and coord_list[
                    prior_proc_base_index * 2 + 1] - b_o <= j:
                    pass

                # The same as source implementation ( similar to fill_quad )
                blkIndexI = i * (blocksize - overlap)
                blkIndexJ = j * (blocksize - overlap)
                refBlockLeft = texture[(blkIndexI):(blkIndexI + blocksize),
                               (blkIndexJ - blocksize + overlap):(blkIndexJ + overlap)]
                refBlockTop = texture[(blkIndexI - blocksize + overlap):(blkIndexI + overlap),
                              (blkIndexJ):(blkIndexJ + blocksize)]

                patchBlock = findPatchBoth(refBlockLeft, refBlockTop, image, blocksize, overlap, tolerance, rng)
                minCutPatch = getMinCutPatchBoth(refBlockLeft, refBlockTop, patchBlock, blocksize, overlap)

                texture[(blkIndexI):(blkIndexI + blocksize), (blkIndexJ):(blkIndexJ + blocksize)] = minCutPatch
            coord_jobs_array[1+sub_job_id] += nW
            if coord_jobs_array[0] > 0:
                break  # if not checked every column, needs to be set to complete to avoid deadlock
        coord_list[pid * 2 + 0] = -1

    Parallel(n_jobs=total_procs, backend="loky", timeout=None)(
        delayed(fill_rows)(i, shm_coord.name, texture_shared_mem_name, job_id + i) for i in range(total_procs))

    shm_coord.close()
    shm_coord.unlink()
    return
