import numpy as np
import cv2


def compute_fft(image):
    image_float = np.float32(image)
    dft = cv2.dft(image_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    return magnitude_spectrum


def compute_freqs_of_interest(spectrum, max_to_fetch: int = 16):
    h, w = spectrum.shape[:2]
    unique_frequencies = set()
    freq_magnitude_pairs = {}

    from math import ceil
    freq_thresh = ceil(min(spectrum.shape[:2])**(1/4))
    print(freq_thresh)

    # Flatten the spectrum and get the indices of the sorted magnitudes
    flat_indices = np.argsort(spectrum, axis=None)[::-1]  # Sort in descending order
    flat_spectrum = spectrum.flatten()

    unique_count = 0

    # Skip the first maximum magnitude
    start_index = 1

    for flat_index in flat_indices[start_index:]:
        if unique_count >= max_to_fetch:
            break

        # Convert flat index to 2D indices
        y, x = np.unravel_index(flat_index, spectrum.shape)
        magnitude = round(flat_spectrum[flat_index])

        # Calculate the frequency as the maximum absolute distance from the center
        freq_y = abs(y - h / 2)
        freq_x = abs(x - w / 2)
        freq = int(max(freq_y, freq_x))
        if freq < freq_thresh:
            continue

        if freq not in unique_frequencies:
            unique_frequencies.add(freq)
            freq_magnitude_pairs[freq] = magnitude
            unique_count += 1
        else:
            if magnitude > freq_magnitude_pairs[freq]:
                freq_magnitude_pairs[freq] = magnitude

    return list(freq_magnitude_pairs.items())


def analyze_freq_spectrum(image, max_components=16):
    magnitude_spectrum = compute_fft(image)
    af = compute_freqs_of_interest(magnitude_spectrum, max_components)
    return af


if __name__ == "__main__":
    image_path = "../t9.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    data = analyze_freq_spectrum(image, max_components=10)
    print(data)
