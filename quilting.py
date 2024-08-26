from .synthesis_subroutines import (
    get_find_patch_to_the_right_method, get_find_patch_below_method, get_find_patch_both_method,
    get_min_cut_patch_horizontal_method, get_min_cut_patch_vertical_method, get_min_cut_patch_both_method)
from multiprocessing.shared_memory import SharedMemory
from dataclasses import dataclass
from .types import UiCoordData, GenParams
from math import ceil
import numpy as np


# -- NOTE -- ___________________________________________________________________________________________________________
#
#   This implementation improves vanilla (a.k.a. sequential) image quilting speed via parallelization.
#
#   The following is a reference to the source implementation contained in the jena2020 folder:
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

# region     methods adapted from jena2020 for re-usability & node compliance

def fill_column(image, initial_block, gen_args: GenParams, rows: int, rng: np.random.Generator):
    find_patch_below = get_find_patch_below_method(gen_args.version)
    get_min_cut_patch = get_min_cut_patch_vertical_method(gen_args.version)
    block_size = initial_block.shape[0]
    overlap = gen_args.overlap
    texture_map = np.zeros(
        ((block_size + rows * (block_size - overlap)), block_size, image.shape[2])).astype(image.dtype)
    texture_map[:block_size, :block_size, :] = initial_block
    for i, blk_idx in enumerate(range((block_size - overlap), texture_map.shape[0] - overlap, (block_size - overlap))):
        ref_block = texture_map[(blk_idx - block_size + overlap):(blk_idx + overlap), :block_size]
        patch_block = find_patch_below(ref_block, image, gen_args, rng)
        min_cut_patch = get_min_cut_patch(ref_block, patch_block, gen_args)
        texture_map[blk_idx:(blk_idx + block_size), :block_size] = min_cut_patch
    return texture_map


def fill_row(image, initial_block, gen_args: GenParams, columns: int, rng: np.random.Generator):
    find_patch_to_the_right = get_find_patch_to_the_right_method(gen_args.version)
    get_min_cut_patch = get_min_cut_patch_horizontal_method(gen_args.version)
    block_size = initial_block.shape[0]
    overlap = gen_args.overlap
    texture_map = np.zeros(
        (block_size, (block_size + columns * (block_size - overlap)), image.shape[2])).astype(image.dtype)
    texture_map[:block_size, :block_size, :] = initial_block
    for i, blk_idx in enumerate(range((block_size - overlap), texture_map.shape[1] - overlap, (block_size - overlap))):
        ref_block = texture_map[:block_size, (blk_idx - block_size + overlap):(blk_idx + overlap)]
        patch_block = find_patch_to_the_right(ref_block, image, gen_args, rng)
        min_cut_patch = get_min_cut_patch(ref_block, patch_block, gen_args)
        texture_map[:block_size, blk_idx:(blk_idx + block_size)] = min_cut_patch
    return texture_map


def fill_quad(rows: int, columns: int, gen_args: GenParams, texture_map, image,
              rng: np.random.Generator, uicd: UiCoordData | None):
    find_patch_both = get_find_patch_both_method(gen_args.version)
    get_min_cut_patch = get_min_cut_patch_both_method(gen_args.version)
    block_size, overlap = gen_args.bo

    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            blk_index_i = i * (block_size - overlap)
            blk_index_j = j * (block_size - overlap)
            ref_block_left = texture_map[
                             blk_index_i:(blk_index_i + block_size),
                             (blk_index_j - block_size + overlap):(blk_index_j + overlap)]
            ref_block_top = texture_map[
                            (blk_index_i - block_size + overlap):(blk_index_i + overlap),
                            blk_index_j:(blk_index_j + block_size)]

            patch_block = find_patch_both(ref_block_left, ref_block_top, image, gen_args, rng)
            min_cut_patch = get_min_cut_patch(ref_block_left, ref_block_top, patch_block, gen_args)

            texture_map[blk_index_i:(blk_index_i + block_size), blk_index_j:(blk_index_j + block_size)] = min_cut_patch

        if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(columns):
            break
    return texture_map


# endregion

# region    parallel solution


def generate_texture_parallel(image, gen_args: GenParams, out_h, out_w, nps,
                              rng: np.random.Generator, uicd: UiCoordData | None):
    """
    @param uicd: contains:
                 * the shared memory name to a one dimensional array that stores the number of "blocks"
                 (with the overlapping area removed) processed by each job;
                 * the job id which should be equal to the batch index, without accounting for the sub processes.
    @param nps: number of parallel stripes; tells how many jobs to use for each of the 4 sections.
    """
    from joblib import Parallel, delayed

    block_size, overlap = gen_args.block_size, gen_args.overlap

    if uicd is not None:
        uicd = UiCoordData(
            uicd.jobs_shm_name,
            uicd.job_id * 4 * nps  # offset job id according to the number of sub jobs
        )

    # Starting block
    source_height, source_width = image.shape[:2]
    corner_y = rng.integers(source_height - block_size)
    corner_x = rng.integers(source_width - block_size)
    start_block = image[corner_y:corner_y + block_size, corner_x:corner_x + block_size]

    # horizontal inverted
    hi_start_block = np.fliplr(start_block)
    hi_image = np.fliplr(image)

    # vertical inverted
    vi_start_block = np.flipud(start_block)
    vi_image = np.flipud(image)

    # note: inverted both ways is computed later when filling the top-left quadrant/section

    # auxiliary variables
    quad_row_width = ceil(out_w / 2 + block_size / 2)
    quad_column_height = ceil(out_h / 2 + block_size / 2)
    cols_per_quad = int(ceil((quad_row_width - block_size) / (block_size - overlap)))  # minding the overlap
    rows_per_quad = int(ceil((quad_column_height - block_size) / (block_size - overlap)))
    # might get 1 more row or column than needed here

    # generate 2 vertical strips and 2 horizontal strips that will split the generated canvas in half
    # the center, where the stripes connect, shares the same tile
    args = [
        (image, start_block, gen_args, cols_per_quad, rng),
        (hi_image, hi_start_block, gen_args, cols_per_quad, rng),
        (image, start_block, gen_args, rows_per_quad, rng),
        (vi_image, vi_start_block, gen_args, rows_per_quad, rng)
    ]
    funcs = [fill_row, fill_row, fill_column, fill_column]
    stripes = Parallel(n_jobs=4, backend="loky", timeout=None)(
        delayed(funcs[i])(*args[i]) for i in range(4))
    hs, his, vs, vis = stripes

    if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(rows_per_quad * 2 + cols_per_quad * 2):
        return None

    # generate the 4 sections (quadrants)
    args = [
        (vis, his, hi_image, rows_per_quad, cols_per_quad, gen_args, nps, rng, uicd),
        (vis, hs, vi_image, rows_per_quad, cols_per_quad, gen_args, nps, rng, uicd),
        (vs, hs, image, rows_per_quad, cols_per_quad, gen_args, nps, rng, uicd),
        (vs, his, hi_image, rows_per_quad, cols_per_quad, gen_args, nps, rng, uicd)
    ]
    funcs = [quad1, quad2, quad3, quad4]
    quads = Parallel(n_jobs=4, backend="loky", timeout=None)(
        delayed(funcs[i])(*args[i]) for i in range(4))
    q1, q2, q3, q4 = quads

    texture = np.zeros((q1.shape[0] * 2 - block_size, q1.shape[1] * 2 - block_size, image.shape[2])).astype(image.dtype)
    bmo = block_size - overlap
    texture[:q1.shape[0] - bmo, :q1.shape[1] - bmo] = q1[:q1.shape[0] - bmo, :q1.shape[1] - bmo]
    texture[:q1.shape[0] - bmo, q1.shape[1] - bmo:] = q2[:q1.shape[0] - bmo, overlap:]
    texture[q1.shape[0] - bmo:, :q1.shape[1] - bmo] = q4[overlap:, :q1.shape[1] - bmo]
    texture[q1.shape[0] - bmo:, q1.shape[1] - bmo:] = q3[overlap:, overlap:]

    return texture[:out_h, :out_w]


def quad1(vis, his, hi_image, rows: int, columns: int, gen_args: GenParams, p_strips, rng,
          uicd: UiCoordData | None):
    """
    @param his: horizontal inverted stripe
    @param vis: vertical inverted stripe
    @param p_strips: number of sub processes to build the quadrant (only when para_lvl higher than 1)
    """
    vi_hi_s = np.ascontiguousarray(np.flipud(his))  # vertical inversion of the horizontal inverted stripe
    hi_vi_s = np.ascontiguousarray(np.fliplr(vis))
    vhi_image = np.ascontiguousarray(np.flipud(hi_image))

    def place_stripes_on_texture():
        texture[:vi_hi_s.shape[0], :vi_hi_s.shape[1]] = vi_hi_s[:, :]
        texture[vi_hi_s.shape[0]:hi_vi_s.shape[0], :hi_vi_s.shape[1]] = hi_vi_s[vi_hi_s.shape[0]:, :]

    if p_strips > 1:
        size = vis.shape[0] * his.shape[1] * hi_image.shape[2] * hi_image.dtype.itemsize
        shm_text = SharedMemory(create=True, size=size)
        try:
            texture = np.ndarray((vis.shape[0], his.shape[1], hi_image.shape[2]), dtype=hi_image.dtype,
                                 buffer=shm_text.buf)
            place_stripes_on_texture()
            fill_quad_ps(rows, columns, gen_args, shm_text.name, vhi_image, p_strips,
                         rng, None if uicd is None else UiCoordData(uicd.jobs_shm_name, uicd.job_id + p_strips * 0))
            texture = np.ascontiguousarray(np.flip(texture, axis=(0, 1)))
        finally:
            shm_text.close()
            shm_text.unlink()
    else:
        texture = np.zeros((vis.shape[0], his.shape[1], hi_image.shape[2])).astype(hi_image.dtype)
        place_stripes_on_texture()
        texture = fill_quad(rows, columns, gen_args, texture, vhi_image, rng,
                            None if uicd is None else UiCoordData(uicd.jobs_shm_name, uicd.job_id + 0))
        texture = np.ascontiguousarray(np.flip(texture, axis=(0, 1)))
    return texture


def quad2(vis, hs, vi_image, rows: int, columns: int, gen_args: GenParams, p_strips, rng,
          uicd: UiCoordData | None):
    vi_hs = np.ascontiguousarray(np.flipud(hs))

    def place_stripes_on_texture():
        texture[:hs.shape[0], :hs.shape[1]] = vi_hs[:, :]
        texture[hs.shape[0]:vis.shape[0], :vis.shape[1]] = vis[hs.shape[0]:, :]

    if p_strips > 1:
        size = vis.shape[0] * hs.shape[1] * vi_image.shape[2] * vi_image.dtype.itemsize
        shm_text = SharedMemory(create=True, size=size)
        try:
            texture = np.ndarray((vis.shape[0], hs.shape[1], vi_image.shape[2]), dtype=vi_image.dtype,
                                 buffer=shm_text.buf)
            place_stripes_on_texture()
            fill_quad_ps(rows, columns, gen_args, shm_text.name, vi_image, p_strips, rng,
                         None if uicd is None else UiCoordData(uicd.jobs_shm_name, uicd.job_id + p_strips * 1))
            texture = np.ascontiguousarray(np.flipud(texture))
        finally:
            shm_text.close()
            shm_text.unlink()
    else:
        texture = np.zeros((vis.shape[0], hs.shape[1], vi_image.shape[2])).astype(vi_image.dtype)
        place_stripes_on_texture()
        texture = fill_quad(rows, columns, gen_args, texture, vi_image, rng,
                            None if uicd is None else UiCoordData(uicd.jobs_shm_name, uicd.job_id + 1))
        texture = np.ascontiguousarray(np.flipud(texture))
    return texture


def quad4(vs, his, hi_image, rows: int, columns: int, gen_args: GenParams, p_strips, rng,
          uicd: UiCoordData | None):
    hi_vs = np.ascontiguousarray(np.fliplr(vs))

    def place_stripes_on_texture():
        texture[:his.shape[0], :his.shape[1]] = his[:, :]
        texture[his.shape[0]:vs.shape[0], :vs.shape[1]] = hi_vs[his.shape[0]:, :]

    if p_strips > 1:
        size = vs.shape[0] * his.shape[1] * hi_image.shape[2] * hi_image.dtype.itemsize
        shm_text = SharedMemory(create=True, size=size)
        try:
            texture = np.ndarray((vs.shape[0], his.shape[1], hi_image.shape[2]), dtype=hi_image.dtype,
                                 buffer=shm_text.buf)
            place_stripes_on_texture()
            fill_quad_ps(rows, columns, gen_args, shm_text.name, hi_image, p_strips, rng,
                         None if uicd is None else UiCoordData(uicd.jobs_shm_name, uicd.job_id + p_strips * 2))
            texture = np.ascontiguousarray(np.fliplr(texture))
        finally:
            shm_text.close()
            shm_text.unlink()
    else:
        texture = np.zeros((vs.shape[0], his.shape[1], hi_image.shape[2])).astype(hi_image.dtype)
        place_stripes_on_texture()
        texture = fill_quad(rows, columns, gen_args, texture, hi_image, rng,
                            None if uicd is None else UiCoordData(uicd.jobs_shm_name, uicd.job_id + 2))
        texture = np.ascontiguousarray(np.fliplr(texture))
    return texture


def quad3(vs, hs, image, rows: int, columns: int, gen_args: GenParams, p_strips, rng,
          uicd: UiCoordData | None):
    def place_stripes_on_texture():
        texture[:hs.shape[0], :hs.shape[1]] = hs[:, :]
        texture[hs.shape[0]:vs.shape[0], :vs.shape[1]] = vs[hs.shape[0]:, :]

    if p_strips > 1:
        size = vs.shape[0] * hs.shape[1] * image.shape[2] * image.dtype.itemsize
        shm_text = SharedMemory(create=True, size=size)
        try:
            texture = np.ndarray((vs.shape[0], hs.shape[1], image.shape[2]), dtype=image.dtype, buffer=shm_text.buf)
            place_stripes_on_texture()
            fill_quad_ps(rows, columns, gen_args, shm_text.name, image, p_strips, rng,
                         None if uicd is None else UiCoordData(uicd.jobs_shm_name, uicd.job_id + p_strips * 3))
            texture = texture.copy()
        finally:
            shm_text.close()
            shm_text.unlink()
    else:
        texture = np.zeros((vs.shape[0], hs.shape[1], image.shape[2])).astype(image.dtype)
        place_stripes_on_texture()
        texture = fill_quad(rows, columns, gen_args, texture, image, rng,
                            None if uicd is None else UiCoordData(uicd.jobs_shm_name, uicd.job_id + 3))
    return texture


# region        parallel cascading stripes

def fill_quad_ps(rows, columns, gen_args: GenParams,
                 texture_shared_mem_name: str, image, total_procs, rng, uicd: UiCoordData | None):
    from joblib import Parallel, delayed
    from multiprocessing import Manager

    # note : ShareableList seems to have problems even though there are no concurrent writes to the same position...
    # setup shared array to store & check each job row & column
    size = 2 * total_procs * np.int32(1).itemsize
    shm_coord = SharedMemory(create=True, size=size)
    try:
        np_coord = np.ndarray((2 * total_procs,), dtype=np.int32, buffer=shm_coord.buf)
        for ip in range(total_procs):
            np_coord[2 * ip] = 1 + ip
            np_coord[2 * ip + 1] = 1

        job_data = ParaRowsJobInfo(
            total_procs=total_procs,
            coord_shared_list_name=shm_coord.name,
            texture_shm_name=texture_shared_mem_name,
            src=image, gen_args=gen_args, rng=rng,
            columns=columns, rows=rows
        )

        with Manager() as manager:
            events = {pid: manager.Event() for pid in range(total_procs)}  # the stripe sub-job index (not the os pid)
            Parallel(n_jobs=total_procs, backend="loky", timeout=None)(
                delayed(fill_rows_ps)(i, job_data, events,
                                      None if uicd is None else UiCoordData(uicd.jobs_shm_name, uicd.job_id + i))
                for i in range(total_procs))
    finally:
        shm_coord.close()
        shm_coord.unlink()


@dataclass
class ParaRowsJobInfo:
    total_procs: int
    coord_shared_list_name: str
    texture_shm_name: str
    src: np.ndarray
    gen_args: GenParams
    rng: np.random.Generator
    columns: int
    rows: int


def fill_rows_ps(pid: int, job: ParaRowsJobInfo, jobs_events: list, uicd: UiCoordData | None):
    gen_args = job.gen_args
    find_patch_both = get_find_patch_both_method(gen_args.version)
    get_min_cut_patch = get_min_cut_patch_both_method(gen_args.version)

    # unwrap data
    block_size, overlap = gen_args.bo
    rng = job.rng
    total_procs, image, rows, columns = job.total_procs, job.src, job.rows, job.columns
    bmo = block_size - overlap
    b_o = ceil(block_size / bmo)

    # get data in shared memory
    shm_coord_ref = SharedMemory(name=job.coord_shared_list_name)
    coord_list = np.ndarray((2 * job.total_procs,), dtype=np.int32, buffer=shm_coord_ref.buf)
    prior_pid = (pid + job.total_procs - 1) % job.total_procs
    shm_texture = SharedMemory(name=job.texture_shm_name)
    texture = np.ndarray((block_size + rows * bmo, block_size + columns * bmo, image.shape[2]), dtype=image.dtype,
                         buffer=shm_texture.buf)

    for i in range(1 + pid, rows + 1, total_procs):
        coord_list[pid * 2 + 1] = -b_o  # indicates no column yet processed on this row
        coord_list[pid * 2 + 0] = i
        for j in range(1, columns + 1):
            # if previous row hasn't processed the adjacent section yet wait for it to advance.
            # -1 is used to shortcircuit this check when the job on the prior row has completed all rows.
            while -1 < coord_list[prior_pid * 2 + 0] < i and \
                    coord_list[prior_pid * 2 + 1] - b_o <= j:
                jobs_events[prior_pid].wait()
                jobs_events[prior_pid].clear()

            # The same as source implementation ( similar to fill_quad )
            blk_index_i = i * bmo
            blk_index_j = j * bmo
            ref_block_left = texture[
                             blk_index_i:(blk_index_i + block_size),
                             (blk_index_j - block_size + overlap):(blk_index_j + overlap)]
            ref_block_top = texture[
                            (blk_index_i - block_size + overlap):(blk_index_i + overlap),
                            blk_index_j:(blk_index_j + block_size)]

            patch_block = find_patch_both(ref_block_left, ref_block_top, image, gen_args, rng)
            min_cut_patch = get_min_cut_patch(ref_block_left, ref_block_top, patch_block, gen_args)

            texture[blk_index_i:(blk_index_i + block_size), blk_index_j:(blk_index_j + block_size)] = min_cut_patch

            coord_list[pid * 2 + 1] = j
            jobs_events[pid].set()

        if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(columns):
            break  # is required to set job as complete and avoid deadlock in jobs waiting for prior row completion

    coord_list[pid * 2 + 0] = -1  # set job as completed
    jobs_events[pid].set()


# endregion     parallel cascading stripes

# endregion     parallel solution

# region    non-parallel solution adapted from jena2020 for node compliance & error function optionality


def generate_texture(image, gen_args: GenParams, out_h, out_w, rng: np.random.Generator,
                     uicd: UiCoordData | None):
    block_size, overlap = gen_args.block_size, gen_args.overlap

    n_h = int(ceil((out_h - block_size) / (block_size - overlap)))
    n_w = int(ceil((out_w - block_size) / (block_size - overlap)))

    texture_map = np.zeros(
        ((block_size + n_h * (block_size - overlap)),
         (block_size + n_w * (block_size - overlap)),
         image.shape[2])).astype(image.dtype)

    # Starting index and block
    h, w = image.shape[:2]
    rand_h = rng.integers(h - block_size)
    rand_w = rng.integers(w - block_size)

    start_block = image[rand_h:rand_h + block_size, rand_w:rand_w + block_size]
    texture_map[:block_size, :block_size, :] = start_block

    # fill 1st row
    texture_map[:block_size, :] = fill_row(image, start_block, gen_args, n_w, rng)
    if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(n_w):
        return None

    # fill 1st column
    texture_map[:, :block_size] = fill_column(image, start_block, gen_args, n_h, rng)
    if uicd is not None and uicd.add_to_job_data_slot_and_check_interrupt(n_h):
        return None

    # fill the rest
    texture_map = fill_quad(n_h, n_w, gen_args, texture_map, image, rng, uicd)

    # crop to final size
    return texture_map[:out_h, :out_w]

# endregion
