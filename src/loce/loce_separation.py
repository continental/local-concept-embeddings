'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

from xai_utils.logging import init_logger
init_logger()

from typing import Iterable

import numpy as np
from sklearn.metrics import pairwise_distances

from .loce import LoCEMultilayerStorageDirectoryLoader


class LoCEMultilayerStorageSeparation(LoCEMultilayerStorageDirectoryLoader):

    def __init__(self,
                 working_directory: str,
                 seed: int = None,
                 min_seg_area: float = 0.0,
                 max_seg_area: float = 1.0
                 ) -> None:
        super().__init__(working_directory, seed, min_seg_area, max_seg_area)

        self.storages = None


    def category_separation(self,
                            retrieval_categories: Iterable[int],
                            retrieval_layer: str) -> None:

        # Load the storages
        self.storages = self.load(retrieval_categories)
        storages_list = [s for _, storages in self.storages.items() for s in storages]

        # Extract labels and points
        labels = np.stack([s.segmentation_category_id for s in storages_list])
        points = np.stack([s.loce_storage[retrieval_layer].loce for s in storages_list])

        # Compute category separation metrics
        categories = np.unique(labels)
        metrics = {}

        for cat in categories:
            # Points in the current category
            cat_points = points[labels == cat]

            # Points not in the current category
            other_points = points[labels != cat]

            # Compute intra-class distance
            intra_distances = pairwise_distances(cat_points)
            intra_dist = np.sum(intra_distances) / (len(cat_points) * (len(cat_points) - 1))

            # Compute inter-class distance
            inter_distances = pairwise_distances(cat_points, other_points)
            inter_dist = np.min(inter_distances)

            # Compute separation index
            separation_index = inter_dist / intra_dist if intra_dist != 0 else float('inf')

            # Store metrics for the category
            metrics[cat] = {
                'intra_class_distance': intra_dist,
                'inter_class_distance': inter_dist,
                'separation_index': separation_index
            }

        return metrics

    def pairwise_category_separation(self,
                                    retrieval_categories: Iterable[int],
                                    retrieval_layer: str) -> np.ndarray:

        # Load the storages
        self.storages = self.load(retrieval_categories)
        storages_list = [s for _, storages in self.storages.items() for s in storages]

        # Extract labels and points
        labels = np.stack([s.segmentation_category_id for s in storages_list])
        points = np.stack([s.loce_storage[retrieval_layer].loce for s in storages_list])

        # Get unique categories and initialize the separation matrix
        categories = np.unique(labels)
        n_categories = len(categories)
        separation_matrix = np.zeros((n_categories, n_categories))

        for i, cat_i in enumerate(categories):
            points_i = points[labels == cat_i]

            for j, cat_j in enumerate(categories):
                if i >= j:  # Avoid duplicates and self-pairs
                    continue

                points_j = points[labels == cat_j]

                # Compute inter-class and intra-class distances
                inter_dist = np.min(pairwise_distances(points_i, points_j))
                combined_points = np.vstack([points_i, points_j])
                intra_dist = np.max(pairwise_distances(combined_points))

                if intra_dist < 1e-6 or inter_dist < 1e-6:
                    separation_matrix[i, j] = 0  # Overlapping or degenerate case
                    separation_matrix[j, i] = 0  # Overlapping or degenerate case

                # Calculate and store the separation index
                sep_index = inter_dist / (intra_dist + 1e-6)
                separation_matrix[i, j] = sep_index
                separation_matrix[j, i] = sep_index  # Symmetric matrix

        return separation_matrix


    def overlap_ratio(self,
                    retrieval_categories: Iterable[int],
                    retrieval_layer: str) -> np.ndarray:
        """
        Compute the overlap ratio matrix for given retrieval categories and a retrieval layer.

        Args:
            retrieval_categories (Iterable[int]): The categories to analyze.
            retrieval_layer (str): The layer of the data to retrieve points.

        Returns:
            np.ndarray: A symmetric matrix where entry (i, j) represents the overlap ratio
                        of category i being "poisoned" by category j.
        """
        # Load the storages
        self.storages = self.load(retrieval_categories)
        storages_list = [s for _, storages in self.storages.items() for s in storages]

        # Extract labels and points
        labels = np.stack([s.segmentation_category_id for s in storages_list])
        points = np.stack([s.loce_storage[retrieval_layer].loce for s in storages_list])

        # Get unique categories and initialize the overlap matrix
        categories = np.unique(labels)
        n_categories = len(categories)
        overlap_matrix = np.zeros((n_categories, n_categories))  # Initialize with zeros

        for i, cat_i in enumerate(categories):
            points_i = points[labels == cat_i]

            for j, cat_j in enumerate(categories):
                if i == j:  # Skip self-category comparison
                    continue

                points_j = points[labels == cat_j]

                # Count points in C_i closer to C_j than to their own points
                self_distances = pairwise_distances(points_i, points_i)
                np.fill_diagonal(self_distances, np.inf)  # Ignore self-distances
                dist_to_self = np.min(self_distances, axis=1)
                dist_to_other = np.min(pairwise_distances(points_i, points_j), axis=1)
                poisoned_points = np.sum(dist_to_other < dist_to_self)

                # Overlap ratio
                overlap = poisoned_points / len(points_i) if len(points_i) > 0 else 0
                overlap_matrix[i, j] = overlap

        return overlap_matrix
