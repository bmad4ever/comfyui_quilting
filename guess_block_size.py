from custom_nodes.comfyui_quilting.misc.bse_desc_util import analyze_keypoint_scales
from custom_nodes.comfyui_quilting.misc.bse_type_aliases import num_pixels, size_weight_pairs
from custom_nodes.comfyui_quilting.misc.bse_ft_util import analyze_freq_spectrum
from math import ceil
import numpy as np


def find_sync_wavelen(pairs: size_weight_pairs, lower, upper) -> num_pixels:
    min_distance_sum = float('inf')
    best_number = lower

    for num in range(lower, upper + 1):
        distance_sum = 0
        for d, w in pairs:
            if num < d:
                distance = d - num
            else:
                remainder = num % d
                distance = min(remainder, d - remainder)

            distance_sum += distance * w

        if distance_sum < min_distance_sum or (abs(distance_sum - min_distance_sum) < .5 and num > best_number):
            min_distance_sum = distance_sum
            best_number = num

    return best_number


def make_guess(pairs: size_weight_pairs, min_dim: num_pixels, max_block_size: num_pixels|None = None) -> num_pixels:
    default_value = round(min_dim / 2.5)  # returned in edge cases

    if len(pairs) == 0:  # an edge case; maybe a blank image is sent...
        return default_value

    block_size_lower_bound = pairs[0][0]
    block_size_upper_bound = round(min_dim / 1.2) if max_block_size is None \
        else min(round(min_dim / 1.2), max_block_size)
    print(f"initial upper bound = {block_size_upper_bound}")

    # the lookup should have at least one single freq sized block of addressable space
    # if this is not the case, analysing a multiple for this freq is irrelevant
    pairs.sort()
    print(pairs)
    while pairs[-1][0] > min_dim - pairs[-1][0]:
        freq, _ = pairs.pop()
        block_size_upper_bound = min(block_size_upper_bound,
                                     min_dim - freq)  # this might be too strong of a restriction
        if len(pairs) == 0:  # edge case
            return default_value

    print(f"rectified upper bound = {block_size_upper_bound}")
    return find_sync_wavelen(pairs, block_size_lower_bound, block_size_upper_bound)


def filter_pairs_by_weight(pairs: size_weight_pairs, weight_percentage_threshold):
    total_weight = sum(weight for _, weight in pairs)
    threshold = total_weight * (weight_percentage_threshold / 100)
    filtered_pairs = [(divisor, weight) for divisor, weight in pairs if weight >= threshold]
    return filtered_pairs


def guess_nice_block_size(src: np.ndarray, freq_analysis_only: bool = False,
                          max_block_size: num_pixels|None = None ) -> num_pixels:
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
    freq_analysis_pairs = analyze_freq_spectrum(src)    # here the image needs to go with float normalized values
    src = (src*255).astype(np.uint8)

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
    freq_analysis_pairs = filter_pairs_by_weight(freq_analysis_pairs[:5], 20)
    desc_analysis_pairs = filter_pairs_by_weight(desc_analysis_pairs[:5], 20)

    final_pairs = [
        *normalize_weights(freq_analysis_pairs),
        *normalize_weights(desc_analysis_pairs)
    ]  # may contain duplicates or multiples, that is expected

    print(f"final pairs: {final_pairs}")
    return make_guess(final_pairs, min_dim, max_block_size)


if __name__ == "__main__":
    from cv2 import imread, IMREAD_GRAYSCALE

    image_path = "diag_wavy.jpg"
    image = imread(image_path, IMREAD_GRAYSCALE)
    block_size = guess_nice_block_size(image)
    print(f"guessed block_size = {block_size}")
    # new values
    # t9  -> 64
    # t16 -> 55 ( the same as prev. )
    # t18 -> 82
    # seem acceptable at a glance
