from .bse_type_aliases import num_pixels, size_weight_pairs
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from collections import Counter
from typing import TypeAlias
import numpy as np
import cv2

area: TypeAlias = int
label: TypeAlias = int


def find_optimal_clusters(data: list, max_k: int = 6) -> tuple[list[label], list[...], int, float]:
    if len(np.unique(data)) == 1:  # edge case
        return [0] * len(data), [data[0]], 1, 0

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
        score = silhouette_score(data, labels, random_state=0)

        if score > best_score:
            best_k = k
            best_score = score
            best_labels = labels
            best_centers = centers

    return best_labels, best_centers.reshape(-1), best_k, best_score


def min_distance_same_label(positions: list[tuple[int, int]], labels: list[label]) -> \
        tuple[dict[label, num_pixels], dict[label, area]]:
    """
    @return: two dictionaries where the keys are the unique labels:
        the 1st contains the minimum distances found for each label;
        the 2nd contains the area between the points from which the minimum distance was obtained.
    """
    positions = np.array(positions)
    labels = np.array(labels)

    unique_labels = np.unique(labels)
    min_distances = {}
    min_dist_areas = {}

    def max_distance(u, v):
        return max(abs(u[0] - v[0]), abs(u[1] - v[1]))

    def area(u, v):
        return abs(u[0] - v[0]) * abs(u[1] - v[1])

    for _label in unique_labels:
        label_indices = np.nonzero(labels == _label)[0]
        if len(label_indices) < 2:
            min_distances[_label] = 0.0  # not enough points to compute distance
            min_dist_areas[_label] = 0.0
            continue

        label_descriptors = positions[label_indices]  # same label descriptors
        distances = pairwise_distances(label_descriptors, label_descriptors, metric=max_distance)
        areas = pairwise_distances(label_descriptors, label_descriptors, metric=area)
        distances[distances < 1] = np.inf  # remove diagonals and less than 1 pixel away pairs

        # find the minimum distance & its area
        min_distance = np.min(distances)
        min_distances[_label] = min_distance
        min_dist_areas[_label] = np.median(areas[distances == min_distance])

    return min_distances, min_dist_areas


def inner_square_area(circle_diameter: float) -> float:
    return 2 * (circle_diameter / 2) ** 2


def analyze_keypoint_scales(image: np.ndarray) -> size_weight_pairs:
    sift = cv2.SIFT_create()
    keypoints = sift.detect(image, None)
    if len(keypoints) == 0:  # edge case
        return []
    kp_sizes = [kp.size for kp in keypoints]  # keypoints' diameters, in pixels
    kp_pts = [kp.pt for kp in keypoints]  # keypoints' (y, x) positions

    # cluster keypoints by size.
    # then, consider their size & distance in the analysis, and weight them w/ respect to area covered
    labels, kp_pt_cluster_centers, _, _ = find_optimal_clusters(kp_sizes)

    # weight clusters with respect to the area coverage on the image
    label_counts = Counter(labels)  # the number of keypoints' of a given label. should be sorted
    labels_coverage = [inner_square_area(diam) * label_counts[i] for i, diam in enumerate(kp_pt_cluster_centers)]
    dist_weight_pairs = [(round(kp_pt_cluster_centers[i]), w) for i, w in enumerate(labels_coverage)]

    # get the minimum distance between keypoints belonging to the same cluster
    # suppose that each keypoints has at least one area adjacent, so that the min. area is given by min_area * num_kps
    distance_pairs, pairs_areas = min_distance_same_label(kp_pts, labels)
    dist_weight_pairs.extend([(round(distance_pairs[i]), pairs_areas[i] * label_counts[i])
                              for i, _ in enumerate(labels_coverage)])
    dist_weight_pairs.sort(key=lambda i: i[1], reverse=True)
    return dist_weight_pairs
