from custom_nodes.comfyui_quilting.misc.bse_desc_util import analyze_keypoint_scales
from custom_nodes.comfyui_quilting.misc.bse_type_aliases import num_pixels, size_weight_pairs
from custom_nodes.comfyui_quilting.misc.bse_ft_util import analyze_freq_spectrum
from math import ceil
import numpy as np
import heapq


def find_sync_wavelens(pairs, lower, upper, n):
    best_values = []  # stores the top n values as (distance_sum, number) tuples

    for num in range(lower, upper + 1):
        distance_sum = 0
        for d, w in pairs:
            if num < d:
                distance = d - num
            else:
                remainder = num % d
                distance = min(remainder, d - remainder)

            distance_sum += distance * w

        heapq.heappush(best_values, (-distance_sum, num))
        if len(best_values) > n:
            heapq.heappop(best_values)

    best_values = [(-ds, num) for ds, num in best_values]  # convert back to positive distance sums
    distances = [val[0] for val in best_values]

    if len(distances) == 0:  # edge case
        return []

    threshold = np.mean(distances) + np.std(distances)
    relevant = [val[1] for val in best_values if val[0] <= threshold]
    return relevant


def make_guess(pairs: size_weight_pairs, min_dim: num_pixels, max_block_size: num_pixels | None = None) -> num_pixels:
    default_value = round(min_dim / 3)  # returned in edge cases

    if len(pairs) == 0:  # an edge case; maybe a blank image is sent...
        return default_value

    # dev note:
    #  it is important to keep at least one block of addressable space for good multiples (where patterns meet).
    #
    #    let "m" be the best multiple, and "b" the potential block size
    #       this implementation tries to ensure that:   min_dim - b >= m
    #    (using a multiple of "m" could help increasing diversity:  min_dim - b >= km, where k is int)
    #
    #  the size of b should fit the size of the biggest repeating pattern,
    #  so that it is not lost due to a small block size.
    #  to select this size is, however, tricky; and it must also not compromise the above condition.
    #
    #    grossly simplifying, this implementation considers that the block size "b" should be equal to "m"
    #    thus, the search upper bound can be obtained by replacing "m" w/ "b":   b < min_dim - b
    #    this allows for a "simple" implementation
    #
    #  unlike previous implementation, pairs with big sizes are not discarded.
    #  these pairs will add weight to higher multiples, skewing the block size toward a high multiple.
    #  since max size is already limited to half min_dim the obtained block size won't be "too big".

    pairs.sort()
    block_size_lower_bound = pairs[0][0]  # the smallest possible distance between a pattern
    block_size_upper_bound = min_dim // 2  # b <= min_dim - b <-> b <= min_dim/2
    if max_block_size is not None and max_block_size < block_size_upper_bound:
        block_size_upper_bound = max_block_size

    rel = find_sync_wavelens(pairs, block_size_lower_bound, block_size_upper_bound,
                             ceil((block_size_upper_bound - block_size_lower_bound) / 10))

    if len(rel) == 0:  # edge case
        return default_value

    return max(rel)  # can afford to go for the max since it won't go over more than half of min_dim


def filter_pairs_by_weight(pairs: size_weight_pairs, weight_percentage_threshold):
    total_weight = sum(weight for _, weight in pairs)
    threshold = total_weight * (weight_percentage_threshold / 100)
    filtered_pairs = [(divisor, weight) for divisor, weight in pairs if weight >= threshold]
    return filtered_pairs


def guess_nice_block_size(src: np.ndarray, freq_analysis_only: bool = False,
                          max_block_size: num_pixels | None = None) -> num_pixels:
    """
    @param src: numpy image with normalized float32 values
    """

    def normalize_weights(pairs: size_weight_pairs):
        if not pairs:
            return []
        weights = [weight for index, weight in pairs]
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_pairs = [(index, weight / total_weight) for index, weight in pairs]
        else:
            normalized_pairs = [(index, 0) for index, weight in pairs]
        return normalized_pairs

    # src should come with normalized float values already
    freq_analysis_pairs = analyze_freq_spectrum(src)  # here the image needs to go with float normalized values
    src = (src * 255).astype(np.uint8)

    # here the image needs to go with integer, 0 to 255, values
    desc_analysis_pairs = [] if freq_analysis_only else analyze_keypoint_scales(src)

    # all pairs should come already sorted in descending order w/ respect to weight
    print(freq_analysis_pairs)
    print(desc_analysis_pairs)

    # filter very small distances, with respect to the src size
    min_dim = min(src.shape[:2])
    thresh_distance = ceil(min_dim ** (1 / 4))
    block_size_upper_bound = round(min_dim / 1.2) if max_block_size is None else max_block_size
    freq_analysis_pairs = [(dst, w) for dst, w in freq_analysis_pairs if
                           thresh_distance <= dst < block_size_upper_bound]
    desc_analysis_pairs = [(dst, w) for dst, w in desc_analysis_pairs if
                           thresh_distance <= dst < block_size_upper_bound]

    # filter distances whose weight is comparatively low
    freq_analysis_pairs = filter_pairs_by_weight(freq_analysis_pairs[:6], 12 if freq_analysis_only else 20)
    desc_analysis_pairs = filter_pairs_by_weight(desc_analysis_pairs[:6], 20)

    final_pairs = [
        *normalize_weights(freq_analysis_pairs),
        *normalize_weights(desc_analysis_pairs)
    ]  # may contain duplicates or multiples, that is expected

    print(f"final pairs: {final_pairs}")
    return make_guess(final_pairs, min_dim, max_block_size)


if __name__ == "__main__":
    from cv2 import imread, IMREAD_GRAYSCALE

    image_path = "t9.png"
    image = imread(image_path, IMREAD_GRAYSCALE)
    block_size = guess_nice_block_size(image, freq_analysis_only=False)
    print(f"guessed block_size = {block_size}")
    # prev values
    # t9  -> 64
    # t16 -> 55
    # t18 -> 82
    # new values  (freq_only=false, true)
    # t9  -> 64, 64
    # t16 -> 44, 44
    # t18 -> 48, 42
    # t166 -> 99, 88  (fixed!)
