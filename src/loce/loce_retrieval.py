'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

from xai_utils.logging import init_logger
init_logger()

from typing import Iterable, Tuple, Literal
import os

import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from .loce import LoCEMultilayerStorageDirectoryLoader, LoCEMultilayerStorage
from .loce_utils import get_rgb_binary_mask, get_colored_mask, get_colored_mask_alt, MSCOCO_CATEGORIES, MSCOCO_CATEGORY_COLORS, shorten_layer_name

from xai_utils.files import mkdir, blend_imgs, add_countours_around_mask


class LoCEMultilayerStorageRetrieval(LoCEMultilayerStorageDirectoryLoader):

    def __init__(self,
                 working_directory: str,
                 seed: int = None,
                 min_seg_area: float = 0.0,
                 max_seg_area: float = 1.0
                 ) -> None:
        super().__init__(working_directory, seed, min_seg_area, max_seg_area)

        self.storages = None

    def _save_one_img(self,
                      pos_samples: Iterable[LoCEMultilayerStorage],
                      neg_samples: Iterable[LoCEMultilayerStorage],
                      pos_samples_dist: Iterable[float],
                      neg_samples_dist: Iterable[float],
                      save_path_base: str,
                      layer: str,
                      pos_samples_cat_idxs: Iterable[int] = None,
                      neg_samples_cat_idxs: Iterable[int] = None,
                      image_tile_size: Tuple[int, int] = (480, 360),
                      frame_width: int = 6
                      ) -> None:
        
        samples = pos_samples + neg_samples
        dists = pos_samples_dist + neg_samples_dist
        if (pos_samples_cat_idxs is None) or (neg_samples_cat_idxs is None):
            frame_colors = ["black"] * len(samples)
        else:
            samples_cat_idxs = pos_samples_cat_idxs + neg_samples_cat_idxs
            frame_colors = ["black"] + ["green" if i == samples_cat_idxs[0] else "red" for i in samples_cat_idxs[1:]]

        titles = ["Query"] + [f"#{i+1} Best" for i in range(len(pos_samples) - 1)] + [f"#{i} Worst" for i in range(len(neg_samples), 0, -1)]

        dist_stings = [f"{d:.3f}" for d in dists]

        titles_with_dists = [f"{t} (dist: {d})" for t, d in zip(titles, dist_stings)]

        save_img_name = f"{samples[0].segmentation_category_id}_{os.path.basename(samples[0].image_path)}.pdf"

        fig, ax = plt.subplots(2, len(samples), figsize=(len(samples) * 2 * 2, 6))

        for col, s in enumerate(samples):
            temp_image_path = s.image_path

            s: LoCEMultilayerStorage = s

            temp_proj = s.loce_storage[layer].projection
            temp_loss = s.loce_storage[layer].loss
            temp_seg = s.segmentation
            temp_cat = s.segmentation_category_id

            temp_color = MSCOCO_CATEGORY_COLORS[temp_cat]
            temp_color_rgb = tuple([int(c * 255) for c in colors.to_rgb(temp_color)])

            rgb_mask = get_rgb_binary_mask(temp_proj)

            img_with_proj = blend_imgs(Image.open(temp_image_path), Image.fromarray(rgb_mask), alpha=0.34)
            temp_proj_resized = np.array(Image.fromarray(temp_proj).resize(img_with_proj.size)) / 255.
            temp_proj_binary = temp_proj_resized > 0.5
            temp_proj_binary_uint8 = temp_proj_binary.astype(np.uint8) * 255
            img_with_seg_counturs = Image.fromarray(add_countours_around_mask(np.array(img_with_proj), temp_proj_binary_uint8, countour_color=(255, 0, 0), thickness=5))
            img_with_proj_resized = np.array(img_with_seg_counturs.resize(image_tile_size))

            img_with_seg = blend_imgs(get_colored_mask_alt(temp_seg, color_channels=[0, 1, 2], mask_value_multipliers=temp_color_rgb), Image.open(temp_image_path), alpha=0.66)
            img_with_seg_counturs = Image.fromarray(add_countours_around_mask(np.array(img_with_seg), temp_seg, countour_color=temp_color_rgb))
            img_with_seg_resized = img_with_seg_counturs.resize(image_tile_size)
            
            for row, img in enumerate([img_with_proj_resized, img_with_seg_resized]):

                ax[row,col].imshow(img)
                ax[row,col].set_xticks([])
                ax[row,col].set_yticks([])

                for spine in ax[row,col].spines.values():
                    spine.set_edgecolor(frame_colors[col])
                    spine.set_linewidth(frame_width)

                if row == 0:
                    ax[row,col].set_title(titles_with_dists[col], fontsize=20)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path_base, save_img_name))
        plt.close()

    @staticmethod
    def _cos_dist_matrix_argsorted(storages: Iterable[LoCEMultilayerStorage],
                                   layer: str
                                   ) -> np.ndarray:

        layer_loces_np = np.array([s.loce_storage[layer].loce for s in storages])

        loce_cos_dist = 1 - cosine_similarity(layer_loces_np)

        sorted_idxs = np.argsort(loce_cos_dist, axis=1)

        return sorted_idxs, loce_cos_dist

    def retrieve_images_vs_one_category(self,
                                        query_category: int,
                                        retrieval_layer: str,
                                        model_tag: str,
                                        img_out_path: str,
                                        retrieve_n_pos_neg_imgs: Tuple[int, int] = (5, 5),
                                        query_img_names: Iterable[str] = None
                                        ) -> None:
        
        save_path_base = f"{img_out_path}/vs_one_category/{model_tag}/{retrieval_layer}"
        
        self.storages = self.load([query_category])
        
        category_storages = self.storages[query_category]

        loce_img_names = [os.path.basename(s.image_path) for s in category_storages]

        if len(category_storages) > 0:

            mkdir(save_path_base)

            sorted_idxs, loce_cos_dist = self._cos_dist_matrix_argsorted(category_storages, retrieval_layer)

            for sample_idx in range(len(category_storages)):

                if not query_img_names is None:
                    if not loce_img_names[sample_idx] in query_img_names:
                        continue

                sorted_idxs_sample = sorted_idxs[sample_idx]

                pos_samples_idxs = sorted_idxs_sample[:retrieve_n_pos_neg_imgs[0]+1] # sample itself (self-distance = 0) + n pos samples
                neg_samples_idxs = sorted_idxs_sample[-retrieve_n_pos_neg_imgs[1]:]

                pos_samples = [category_storages[i] for i in pos_samples_idxs]
                neg_samples = [category_storages[i] for i in neg_samples_idxs]

                pos_samples_dist = [loce_cos_dist[sample_idx,i] for i in pos_samples_idxs]
                neg_samples_dist = [loce_cos_dist[sample_idx,i] for i in neg_samples_idxs]

                self._save_one_img(pos_samples, neg_samples, pos_samples_dist, neg_samples_dist, save_path_base, retrieval_layer)

        else:
            print("No storages of given category")

    def retrieve_images_vs_selected_categories(self,
                                               query_img_names: Iterable[str],
                                               query_img_category_idxs: Iterable[int],
                                               retrieve_from_categories: Iterable[int],
                                               retrieval_layer: str,
                                               model_tag: str,
                                               img_out_path: str,
                                               n_pos_neg_imgs: Tuple[int, int] = (5, 5)
                                               ) -> None:
        
        save_path_base = f"{img_out_path}/vs_selected_categories/{model_tag}/{retrieval_layer}"
        
        self.storages = self.load(retrieve_from_categories)

        storages_list = [s for _, storages in self.storages.items() for s in storages]

        storage_img_names_category_idxs_pairs = [(s.segmentation_category_id, os.path.basename(s.image_path)) for s in storages_list]

        query_img_name_category_idx_pairs = [(cat_idx, img_name) for img_name, cat_idx in zip(query_img_names, query_img_category_idxs)]

        if len(storages_list) > 0:

            mkdir(save_path_base)

            sorted_idxs, loce_cos_dist = self._cos_dist_matrix_argsorted(storages_list, retrieval_layer)

            for sample_idx in range(len(storages_list)):

                storage_img_name_category_idx_pair = storage_img_names_category_idxs_pairs[sample_idx]

                if not storage_img_name_category_idx_pair in query_img_name_category_idx_pairs:
                    continue
                
                sorted_idxs_sample = sorted_idxs[sample_idx]

                pos_samples_idxs = sorted_idxs_sample[:n_pos_neg_imgs[0]+1] # sample itself (self-distance = 0) + n pos samples
                neg_samples_idxs = sorted_idxs_sample[-n_pos_neg_imgs[1]:]

                pos_samples = [storages_list[i] for i in pos_samples_idxs]
                neg_samples = [storages_list[i] for i in neg_samples_idxs]

                pos_samples_dist = [loce_cos_dist[sample_idx,i] for i in pos_samples_idxs]
                neg_samples_dist = [loce_cos_dist[sample_idx,i] for i in neg_samples_idxs]

                pos_samples_cat_idxs = [storage_img_names_category_idxs_pairs[i][0] for i in pos_samples_idxs]
                neg_samples_cat_idxs = [storage_img_names_category_idxs_pairs[i][0] for i in neg_samples_idxs]

                self._save_one_img(pos_samples, neg_samples, pos_samples_dist, neg_samples_dist, save_path_base, retrieval_layer, pos_samples_cat_idxs, neg_samples_cat_idxs)

        else:
            print("No storages of given category")

    def retrieve_outliers(self,
                          categories: Iterable[str],
                          retrieval_layer: str,
                          model_tag: str,
                          img_out_path: str,
                          mode: Literal["distance", "lof", "isolation"] = "distance",
                          n_samples: int = 7
                          ) -> None:
        
        save_path_base = f"{img_out_path}/outliers_{mode}/"
        mkdir(save_path_base)
        
        self.storages = self.load(categories)

        for category_idx, category_storages in self.storages.items():

            if len(category_storages) > 0:

                if mode == "lof":
                    layer_loces_np = np.array([s.loce_storage[retrieval_layer].loce for s in category_storages])

                    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, metric="cosine")
                    y_pred = lof.fit_predict(layer_loces_np)

                    outlier_indices = np.where(y_pred == -1)[0]
                    outlier_scores = lof.negative_outlier_factor_[y_pred == -1]
                    sorted_outlier_indices = np.argsort(outlier_scores)

                    sorted_metrics = [outlier_scores[idx] for idx in sorted_outlier_indices]
                    sorted_storages = [category_storages[idx] for idx in outlier_indices[np.argsort(outlier_scores)]]

                elif mode == "isolation":
                    layer_loces_np = np.array([s.loce_storage[retrieval_layer].loce for s in category_storages])

                    clf = IsolationForest(contamination=0.05)
                    y_pred = clf.fit_predict(layer_loces_np)
                    anomaly_scores = clf.decision_function(layer_loces_np)

                    outlier_indices = np.where(y_pred == -1)[0]
                    outlier_scores = anomaly_scores[y_pred == -1]
                    sorted_outlier_indices = np.argsort(outlier_scores)

                    sorted_metrics = [outlier_scores[idx] for idx in sorted_outlier_indices]
                    sorted_storages = [category_storages[idx] for idx in outlier_indices[np.argsort(outlier_scores)]]

                else:
                    _, loce_cos_dist = self._cos_dist_matrix_argsorted(category_storages, retrieval_layer)

                    distance_matrix = loce_cos_dist

                    #if mode == "rank":
                    #    loce_orders = np.argsort(loce_cos_dist, axis=1)
                    #    loce_ranks = np.argsort(loce_orders, axis=1) + 1
                    #    distance_matrix = loce_ranks

                    # outliers in the end of this list
                    distances = distance_matrix.sum(axis=1)
                    argsorted_cumulative_distances = np.argsort(distances)
                    sorted_metrics = [distances[idx] for idx in argsorted_cumulative_distances]
                    sorted_storages = [category_storages[idx] for idx in argsorted_cumulative_distances]

                self._save_one_img_retrieval(sorted_storages[-n_samples:], sorted_metrics[-n_samples:], save_path_base, model_tag, retrieval_layer, category_idx, mode)
                #save_storages_as_tiles_together(sorted_storages, save_path_base, model_tag, category_idx, layers=[retrieval_layer])

            else:
                print(f"No storages of given category: {category_idx}")

    def _save_one_img_retrieval(self,
                                samples: Iterable[LoCEMultilayerStorage],
                                dists: Iterable[float],
                                save_path_base: str,
                                model: str,
                                layer: str,
                                category_idx: int,
                                metric_tag: str,
                                image_tile_size: Tuple[int, int] = (480, 360)
                                ) -> None:
    

        titles = [f"#{i} Outlier" for i in range(len(samples), 0, -1)]

        dist_stings = [f"{d:.3f}" for d in dists]

        titles_with_dists = [f"{t} ({metric_tag}: {d})" for t, d in zip(titles, dist_stings)]

        save_img_name = f"{MSCOCO_CATEGORIES[category_idx]}_{model}_{layer}.pdf"

        fig, ax = plt.subplots(2, len(samples), figsize=(len(samples) * 2 * 2, 6))

        for col, s in enumerate(samples):
            temp_image_path = s.image_path

            temp_proj = s.loce_storage[layer].projection
            temp_loss = s.loce_storage[layer].loss
            temp_seg = s.segmentation

            rgb_mask = get_rgb_binary_mask(temp_proj)

            img_with_proj = blend_imgs(Image.open(temp_image_path), Image.fromarray(rgb_mask), alpha=0.34)
            temp_proj_resized = np.array(Image.fromarray(temp_proj).resize(img_with_proj.size)) / 255.
            temp_proj_binary = temp_proj_resized > 0.5
            temp_proj_binary_uint8 = temp_proj_binary.astype(np.uint8) * 255
            img_with_seg_counturs = Image.fromarray(add_countours_around_mask(np.array(img_with_proj), temp_proj_binary_uint8, countour_color=(255, 0, 0), thickness=5))
            img_with_proj_resized = np.array(img_with_seg_counturs.resize(image_tile_size))

            img_with_seg = blend_imgs(get_colored_mask(temp_seg, mask_value_multiplier=255), Image.open(temp_image_path), alpha=0.66)
            img_with_seg_counturs = Image.fromarray(add_countours_around_mask(np.array(img_with_seg), temp_seg))
            img_with_seg_resized = img_with_seg_counturs.resize(image_tile_size)
            
            for row, img in enumerate([img_with_proj_resized, img_with_seg_resized]):

                ax[row,col].imshow(img)
                ax[row,col].axis('off')

                if row == 0:
                    ax[row,col].set_title(titles_with_dists[col], fontsize=20)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path_base, save_img_name))
        plt.close()

    def retrieval_mAP(self,
                      retrieval_categories: Iterable[int],
                      retrieval_layer: str,
                      mAP_at_n: Iterable[int] = [1, 5, 10, 20, 30, 40],
                      ) -> None:
        
        def calculate_precision(elements, ground_truth_label):
            match_count = sum(1 for element in elements if element == ground_truth_label)
            
            precision = match_count / len(elements) if elements else 0
            
            return precision

        self.storages = self.load(retrieval_categories)

        storages_list = [s for _, storages in self.storages.items() for s in storages]

        results = {c: {n: [] for n in mAP_at_n} for c in retrieval_categories}

        if len(storages_list) > 0:

            sorted_idxs, loce_cos_dist = self._cos_dist_matrix_argsorted(storages_list, retrieval_layer)

            for sample_idx in range(len(storages_list)):

                sorted_idxs_sample = sorted_idxs[sample_idx]

                for n in mAP_at_n:

                    topn_retrieved_idxs_sample = sorted_idxs_sample[:n+1]

                    topn_retrieved_labels_sample = [storages_list[i].segmentation_category_id for i in topn_retrieved_idxs_sample]

                    query_label = topn_retrieved_labels_sample[0]
                    
                    retrieved_elements_labels = topn_retrieved_labels_sample[1:]

                    precision = calculate_precision(retrieved_elements_labels, query_label)

                    results[query_label][n].append(precision)
    
        else:
            print("No storages of given categories")

        results = {c: {n: np.array(results[c][n]) for n in mAP_at_n} for c in retrieval_categories}

        return results

    def retrieve_images_vs_selected_categories_one_plot(self,
                                                        axes: Iterable[Iterable[plt.Axes]],  # plotting axes
                                                        query_img_names: Iterable[str],
                                                        query_img_category_idxs: Iterable[int],
                                                        retrieve_from_categories: Iterable[int],
                                                        retrieval_layer: str,
                                                        model_tag: str,
                                                        n_pos_samples: int = 5,
                                                        show_lloce: bool = False,  # show original mask otherwise,
                                                        image_tile_size: Tuple[int, int] = (240, 180),
                                                        ) -> None:
        
        self.storages = self.load(retrieve_from_categories)

        storages_list = [s for _, storages in self.storages.items() for s in storages]

        storage_img_names_category_idxs_pairs = [(s.segmentation_category_id, os.path.basename(s.image_path)) for s in storages_list]

        query_img_name_category_idx_pairs = [(cat_idx, img_name) for img_name, cat_idx in zip(query_img_names, query_img_category_idxs)]

        axes_idx = 0

        if len(storages_list) > 0:

            sorted_idxs, loce_cos_dist = self._cos_dist_matrix_argsorted(storages_list, retrieval_layer)

            for sample_idx in range(len(storages_list)):

                storage_img_name_category_idx_pair = storage_img_names_category_idxs_pairs[sample_idx]

                if not storage_img_name_category_idx_pair in query_img_name_category_idx_pairs:
                    continue
                
                sorted_idxs_sample = sorted_idxs[sample_idx]

                pos_samples_idxs = sorted_idxs_sample[:n_pos_samples+1] # sample itself (self-distance = 0) + n pos samples

                pos_samples = [storages_list[i] for i in pos_samples_idxs]

                pos_samples_dist = [loce_cos_dist[sample_idx,i] for i in pos_samples_idxs]

                pos_samples_cat_idxs = [storage_img_names_category_idxs_pairs[i][0] for i in pos_samples_idxs]

                
                axes[axes_idx][0].text(0.5, 0.5, f"{model_tag}\n{shorten_layer_name(retrieval_layer)}", ha='center', va='center', fontsize=30)
                axes[axes_idx][0].axis('off')
                self._save_one_img_on_axes(axes[axes_idx][1:], pos_samples, pos_samples_dist, retrieval_layer, pos_samples_cat_idxs, image_tile_size=image_tile_size, show_lloce=show_lloce)
                axes_idx += 1

        else:
            print("No storages of given category")

    def _save_one_img_on_axes(self,
                              axes: Iterable[plt.Axes],  # plotting axes
                              pos_samples: Iterable[LoCEMultilayerStorage],
                              pos_samples_dist: Iterable[float],
                              layer: str,
                              pos_samples_cat_idxs: Iterable[int] = None,
                              image_tile_size: Tuple[int, int] = (240, 180),
                              frame_width: int = 5,
                              show_lloce: bool = False,
                              titles_outliers: bool = False
                              ) -> None:
        
        samples = pos_samples
        dists = pos_samples_dist
        if pos_samples_cat_idxs is None:
            frame_colors = ["black"] * len(samples)
        else:
            samples_cat_idxs = pos_samples_cat_idxs
            frame_colors = ["black"] + ["green" if i == samples_cat_idxs[0] else "red" for i in samples_cat_idxs[1:]]

        dist_stings = [f"{d:.3f}" for d in dists]

        if titles_outliers:
            titles_with_dists = [rf"$\Sigma L_2$: {d}" for d in dist_stings]
        else:
            titles = ["Query"] + [f"#{i+1}" for i in range(len(samples) - 1)]
            #titles_with_dists = [f"{t}, $L_2$: {d}" for t, d in zip(titles, dist_stings)]
            titles_with_dists = [f"$L_2$: {d}" for t, d in zip(titles, dist_stings)]
            titles_with_dists[0] = "Query"

        save_img_name = f"{samples[0].segmentation_category_id}_{os.path.basename(samples[0].image_path)}.pdf"

        for col, (s, ax) in enumerate(zip(samples, axes)):
            temp_image_path = s.image_path

            s: LoCEMultilayerStorage = s

            temp_proj = s.loce_storage[layer].projection
            temp_loss = s.loce_storage[layer].loss
            temp_seg = s.segmentation
            temp_cat = s.segmentation_category_id

            temp_color = MSCOCO_CATEGORY_COLORS[temp_cat]
            temp_color_rgb = tuple([int(c * 255) for c in colors.to_rgb(temp_color)])

            rgb_mask = get_rgb_binary_mask(temp_proj)

            img_with_proj = blend_imgs(Image.open(temp_image_path), Image.fromarray(rgb_mask), alpha=0.34)
            temp_proj_resized = np.array(Image.fromarray(temp_proj).resize(img_with_proj.size)) / 255.
            temp_proj_binary = temp_proj_resized > 0.5
            temp_proj_binary_uint8 = temp_proj_binary.astype(np.uint8) * 255
            img_with_seg_counturs = Image.fromarray(add_countours_around_mask(np.array(img_with_proj), temp_proj_binary_uint8, countour_color=(255, 0, 0), thickness=5))
            img_with_proj_resized = img_with_seg_counturs.resize(image_tile_size)

            img_with_seg = blend_imgs(get_colored_mask_alt(temp_seg, color_channels=[0, 1, 2], mask_value_multipliers=temp_color_rgb), Image.open(temp_image_path), alpha=0.66)
            img_with_seg_counturs = Image.fromarray(add_countours_around_mask(np.array(img_with_seg), temp_seg, countour_color=temp_color_rgb))
            img_with_seg_resized = img_with_seg_counturs.resize(image_tile_size)
            
            if show_lloce:
                img = img_with_proj_resized
            else:
                img = img_with_seg_resized

            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_edgecolor(frame_colors[col])
                spine.set_linewidth(frame_width)

            ax.text(1, 1, titles_with_dists[col], fontsize=24, color='black',
                    verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'),
                    transform=ax.transAxes)
            #ax.set_title(titles_with_dists[col], fontsize=22)


    def retrieve_outliers_one_plot(self,
                                   axes: Iterable[plt.Axes],
                                   categories: Iterable[str],
                                   retrieval_layer: str,
                                   model_tag: str,
                                   n_samples: int = 7,
                                   show_lloce: bool = False,
                                   image_tile_size: Tuple[int, int] = (240, 180),
                                   ) -> None:
        
        self.storages = self.load(categories)

        axes_idx = 0

        for category_idx, category_storages in self.storages.items():

            if len(category_storages) > 0:

                _, loce_cos_dist = self._cos_dist_matrix_argsorted(category_storages, retrieval_layer)

                distance_matrix = loce_cos_dist

                distances = distance_matrix.sum(axis=1)
                argsorted_cumulative_distances = np.argsort(distances)
                sorted_metrics = [distances[idx] for idx in argsorted_cumulative_distances]
                sorted_storages = [category_storages[idx] for idx in argsorted_cumulative_distances]

                axes[axes_idx][0].text(0.5, 0.5, f"{model_tag}\n{shorten_layer_name(retrieval_layer)}", ha='center', va='center', fontsize=30)
                axes[axes_idx][0].axis('off')
                self._save_one_img_on_axes(axes[axes_idx][1:][::-1], sorted_storages[-n_samples:], sorted_metrics[-n_samples:], retrieval_layer, None, image_tile_size=image_tile_size, frame_width=0, show_lloce=show_lloce, titles_outliers=True)
                axes_idx += 1

            else:
                print("No storages of given categories")


    def _save_one_img_retrieval_one_plot(self,
                                         samples: Iterable[LoCEMultilayerStorage],
                                         dists: Iterable[float],
                                         save_path_base: str,
                                         model: str,
                                         layer: str,
                                         category_idx: int,
                                         metric_tag: str,
                                         image_tile_size: Tuple[int, int] = (480, 360)
                                         ) -> None:
    

        titles = [f"#{i} Outlier" for i in range(len(samples), 0, -1)]

        dist_stings = [f"{d:.3f}" for d in dists]

        titles_with_dists = [f"{t} ({metric_tag}: {d})" for t, d in zip(titles, dist_stings)]

        save_img_name = f"{MSCOCO_CATEGORIES[category_idx]}_{model}_{layer}.pdf"

        fig, ax = plt.subplots(2, len(samples), figsize=(len(samples) * 2 * 2, 6))

        for col, s in enumerate(samples):
            temp_image_path = s.image_path

            temp_proj = s.loce_storage[layer].projection
            temp_loss = s.loce_storage[layer].loss
            temp_seg = s.segmentation

            rgb_mask = get_rgb_binary_mask(temp_proj)

            img_with_proj = blend_imgs(Image.open(temp_image_path), Image.fromarray(rgb_mask), alpha=0.34)
            temp_proj_resized = np.array(Image.fromarray(temp_proj).resize(img_with_proj.size)) / 255.
            temp_proj_binary = temp_proj_resized > 0.5
            temp_proj_binary_uint8 = temp_proj_binary.astype(np.uint8) * 255
            img_with_seg_counturs = Image.fromarray(add_countours_around_mask(np.array(img_with_proj), temp_proj_binary_uint8, countour_color=(255, 0, 0), thickness=5))
            img_with_proj_resized = np.array(img_with_seg_counturs.resize(image_tile_size))

            img_with_seg = blend_imgs(get_colored_mask(temp_seg, mask_value_multiplier=255), Image.open(temp_image_path), alpha=0.66)
            img_with_seg_counturs = Image.fromarray(add_countours_around_mask(np.array(img_with_seg), temp_seg))
            img_with_seg_resized = img_with_seg_counturs.resize(image_tile_size)
            
            for row, img in enumerate([img_with_proj_resized, img_with_seg_resized]):

                ax[row,col].imshow(img)
                ax[row,col].axis('off')

                if row == 0:
                    ax[row,col].set_title(titles_with_dists[col], fontsize=20)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path_base, save_img_name))
        plt.close()
