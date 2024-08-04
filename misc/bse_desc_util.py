from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from collections import Counter
from typing import TypeAlias
import numpy as np
import cv2

num_pixels: TypeAlias = int
weight: TypeAlias = float


def find_optimal_clusters(data: list, max_k: int = 6) -> tuple[list[int], list[...], int, float]:
    iters = range(2, max_k + 1)
    best_k = 2
    best_score = -1.0
    best_labels = []
    best_centers = []
    data = np.array(data).reshape(-1, 1)

    for k in iters:
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0)
        kmeans.fit(data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        score = silhouette_score(data, labels)

        if score > best_score:
            best_k = k
            best_score = score
            best_labels = labels
            best_centers = centers

    return best_labels, best_centers.reshape(-1), best_k, best_score


def min_distance_same_label(positions: list[tuple[int, int]], labels: list[int]) -> \
        tuple[dict[num_pixels, int], dict[num_pixels, int]]:
    positions = np.array(positions)
    labels = np.array(labels)

    unique_labels = np.unique(labels)
    min_distances = {}
    min_dist_areas = {}

    def max_distance(u, v):
        return max(abs(u[0] - v[0]), abs(u[1] - v[1]))

    def area(u, v):
        return abs(u[0] - v[0]) * abs(u[1] - v[1])

    for label in unique_labels:
        label_indices = np.nonzero(labels == label)[0]
        if len(label_indices) < 2:
            min_distances[label] = None  # not enough points to compute distance
            continue

        label_descriptors = positions[label_indices]  # same label descriptors
        distances = pairwise_distances(label_descriptors, label_descriptors, metric=max_distance)
        areas = pairwise_distances(label_descriptors, label_descriptors, metric=area)
        distances[distances < 1] = np.inf  # remove diagonals and less than 1 pixel away pairs

        # find the minimum distance & its area
        min_distance = np.min(distances)
        min_distances[label] = min_distance
        min_dist_areas[label] = np.median(areas[distances == min_distance])

    return min_distances, min_dist_areas


def inner_square_area(circle_diameter: float) -> float:
    return 2*(circle_diameter/2)**2


def analyze_keypoint_scales(image: np.ndarray) -> list[tuple[num_pixels, weight]]:
    sift = cv2.SIFT_create()
    keypoints = sift.detect(image, None)
    kp_sizes = [kp.size for kp in keypoints]  # keypoints' diameters, in pixels
    kp_pts = [kp.pt for kp in keypoints]      # keypoints' (y, x) positions

    # cluster keypoints by size.
    # then, consider their size & distance in the analysis, and weight them w/ respect to area covered
    labels, kp_pt_cluster_centers, _, _ = find_optimal_clusters(kp_sizes)

    # weight clusters with respect to the area coverage on the image
    label_counts = Counter(labels)  # the number of keypoints' of a given label. should be sorted
    labels_coverage = [inner_square_area(diam)*label_counts[i] for i, diam in enumerate(kp_pt_cluster_centers)]
    dist_weight_pairs = [(round(kp_pt_cluster_centers[i]), w) for i, w in enumerate(labels_coverage)]

    # get the minimum distance between keypoints belonging to the same cluster
    # suppose that each keypoints has at least one area adjacent, so that the min. area is given by min_area * num_kps
    distance_pairs, pairs_areas = min_distance_same_label(kp_pts, labels)
    dist_weight_pairs.extend([(round(distance_pairs[i]), pairs_areas[i]*label_counts[i])
                              for i, _ in enumerate(labels_coverage)])
    dist_weight_pairs.sort(key=lambda i: i[1], reverse=True)
    print(f"sorted descs = {dist_weight_pairs}")
    return dist_weight_pairs


if __name__ == "__main__":
    image_path = '../t18.png'
    image = cv2.imread(image_path)
    data = analyze_keypoint_scales(image)
    print(data)