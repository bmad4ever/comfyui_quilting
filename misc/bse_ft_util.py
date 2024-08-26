from .bse_type_aliases import size_weight_pairs
import numpy as np
import cv2


def compute_fft(image):
    image = np.float32(image)
    dft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    return magnitude_spectrum


def compute_wavelens_of_interest(spectrum: np.ndarray, max_to_fetch: int = 16) -> size_weight_pairs:
    h, w = spectrum.shape[:2]
    unique_wavelen = set()
    wavelen_magnitude_pairs = {}

    # flatten the spectrum and get the indices of the sorted magnitudes
    flat_indices = np.argsort(spectrum, axis=None)[::-1]  # sort in descending order
    flat_spectrum = spectrum.flatten()

    unique_count = 0

    start_index = 1  # skip the first maximum magnitude
    for flat_index in flat_indices[start_index:]:
        if unique_count >= max_to_fetch:
            break

        # convert flat index to 2D indices
        y, x = np.unravel_index(flat_index, spectrum.shape)
        magnitude = round(flat_spectrum[flat_index])

        # calculate the frequency as the maximum absolute distance from the center
        freq_y = abs(y - h / 2) / h
        freq_x = abs(x - w / 2) / w
        # compute wavelen
        wavelen_y = 1 / freq_y if freq_y > 0 else 0  # don't return infinity when selecting max
        wavelen_x = 1 / freq_x if freq_x > 0 else 0
        wavelen = int(max(wavelen_y, wavelen_x))

        if wavelen not in unique_wavelen:
            unique_wavelen.add(wavelen)
            wavelen_magnitude_pairs[wavelen] = magnitude
            unique_count += 1
        else:
            if magnitude > wavelen_magnitude_pairs[wavelen]:
                wavelen_magnitude_pairs[wavelen] = magnitude

    return list(wavelen_magnitude_pairs.items())


def analyze_freq_spectrum(image: np.ndarray, max_items: int = 16) -> size_weight_pairs:
    magnitude_spectrum = compute_fft(image)
    wlen_mag_pairs = compute_wavelens_of_interest(magnitude_spectrum, max_items)
    return wlen_mag_pairs
