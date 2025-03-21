'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''


import sys
import os
sys.path.append(os.getcwd())
from xai_utils.logging import init_logger
init_logger()

import random
from typing import Tuple, Iterable, Dict, List, Literal, Any
import math

import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
from matplotlib.patheffects import withStroke
from PIL import Image, ImageDraw
from tqdm import tqdm

from xai_utils.files import mkdir, rmdir, blend_imgs, add_countours_around_mask, write_pickle
from .loce import LoCEMultilayerStorage, LoCEMultilayerStorageDirectoryLoader
from .loce_experiment_constants import LoCEExperimentConstants
from .loce_utils import draw_mscoco_categories_and_colors, draw_voc_categories_and_colors, find_closest_square_rootable_number, get_colored_mask_alt, draw_frame, MSCOCO_MARKERS, VOC_MARKERS
from .loce_clusterer import LoCEClusterer


class LoCEDimensionReducer:

    category_sets = {
        #"animals_capy": LoCEExperimentConstants.MSCOCO_CATEGORIES_ANIMALS_CAPY,
        #"all_capy": LoCEExperimentConstants.MSCOCO_CATEGORIES_ALL_CAPY,
        "all": LoCEExperimentConstants.MSCOCO_CATEGORIES_ALL,    
        #"vehicles": LoCEExperimentConstants.MSCOCO_CATEGORIES_VEHICLES,
        #"animals": LoCEExperimentConstants.MSCOCO_CATEGORIES_ANIMALS,    
        #"car_bus_truck": LoCEExperimentConstants.MSCOCO_CATEGORIES_CAR_BUS_TRUCK,
    }

    category_sets_purity = {
        #"animals_capy": LoCEExperimentConstants.MSCOCO_CATEGORIES_ANIMALS_CAPY,
        #"all_capy": LoCEExperimentConstants.MSCOCO_CATEGORIES_ALL_CAPY,
        "all": LoCEExperimentConstants.MSCOCO_CATEGORIES_ALL,    
        #"vehicles": LoCEExperimentConstants.MSCOCO_CATEGORIES_VEHICLES,
        #"animals": LoCEExperimentConstants.MSCOCO_CATEGORIES_ANIMALS
    }

    category_sets_voc = {"all_voc": LoCEExperimentConstants.VOC_CATEGORIES}

    category_sets_purity_voc = {"all_voc": LoCEExperimentConstants.VOC_CATEGORIES}

    base_alpha = 0.9

    # alphas: (alpha_points, alpha_gmm, alpha_mesh)
    plot_2d_kwargs = {
        "points": (base_alpha, 0.0, 0.0),
        "gmms": (0.0, base_alpha, 0.0),
        "densities": (0.0, 0.0, base_alpha),
        "points_gmms": (base_alpha/2, base_alpha, 0.0),
        "points_densities": (base_alpha, 0.0, base_alpha/2),
        "gmms_densities": (0.0, base_alpha, base_alpha/2),
        "points_gmms_densities": (base_alpha/2, base_alpha, base_alpha/4),
    }

    plot_2d_kwargs_4 = {
        "points_gmms": (base_alpha/2, base_alpha, 0.0),
        "points_densities": (base_alpha, 0.0, base_alpha/2),
        "gmms_densities": (0.0, base_alpha, base_alpha/2),
        "points_gmms_densities": (base_alpha/2, base_alpha, base_alpha/4),
    }

    plot_2d_kwargs_3 = {
        "points_densities": (base_alpha, 0.0, base_alpha/2),
        "gmms_densities": (0.0, base_alpha, base_alpha/2),
        "points_gmms_densities": (base_alpha/2, base_alpha, base_alpha/4),
    }

    plot_2d_kwargs_3_grey = {
        "points_densities": (base_alpha, 0.0, base_alpha/2),
        "gmms_densities": (0.0, base_alpha, base_alpha/2),
        "points_gmms_densities": (base_alpha/2, base_alpha, base_alpha/2),
    }

    plot_2d_kwargs_2 = {
        "points_densities": (base_alpha, 0.0, base_alpha/2),
        "gmms_densities": (0.0, base_alpha, base_alpha/2),
    }

    plot_2d_kwargs_1 = {
        "points_gmms_densities": (base_alpha/2, base_alpha, base_alpha/4),
    }

    def __init__(self,
                 clustering_settings: str = 'relaxed',
                 segmenter_tag: str = 'original',
                 marker_size: int = 30,
                 seed: int = 42,
                 in_dir_imgs: str = f"./data/mscoco2017val/val2017/",
                 n_samples_per_tag: int = 100,
                 method_distance: Iterable[Tuple[str, str]] = [("ward", "euclidean")],  # [("ward", "euclidean"), ("complete", "cosine")],
                 out_dir_base = f'./experiment_outputs/umap/'
                 ) -> None:
        
        self.clustering_settings = clustering_settings
        self.segmenter_tag = segmenter_tag
        self.marker_size = marker_size
        self.seed = seed
        self.in_dir_imgs = in_dir_imgs
        self.n_samples_per_tag = n_samples_per_tag
        self.method_distance = method_distance

        self.cluster_purity = LoCEExperimentConstants.PURITY[clustering_settings],
        self.sample_threshold_coefficient = LoCEExperimentConstants.SAMPLE_THRESHOLD_COEFFICIENT[clustering_settings]
        random.seed(self.seed)

        self.out_dir_base = out_dir_base

        self.alpha_kwargs_current = self.plot_2d_kwargs_3
        self.alpha_kwargs_grey_current = self.plot_2d_kwargs_3_grey

    def _get_info_from_storages(self, 
                                storages: Dict[int, List[LoCEMultilayerStorage]],
                                layer: str):
        storages_flat = [item for sublist in storages.values() for item in sublist[:self.n_samples_per_tag]]
        labels = np.array([s.segmentation_category_id for s in storages_flat])
        loces = np.array([s.get_loce(layer).loce for s in storages_flat])
        return loces, labels, storages_flat

    @staticmethod
    def get_gmm_populations(gmm, umap2d_result_class):
        gmm_preds = np.argmax(gmm.predict_proba(umap2d_result_class), axis=1)
        n_class_samples = len(gmm_preds)
        uniques, counts = np.unique(gmm_preds, return_counts=True)
        component_populations = counts / n_class_samples
        return component_populations

    @staticmethod
    def _fit_gmms_without_outliers(umap2d_result: np.ndarray,
                                   labels: np.ndarray,
                                   max_components: int = 5,
                                   ignore_gmm_weight_threshold: float = 0.1):
        
        unique_labels = np.unique(labels)

        gmms = {}
        filtered_gmms = {}
        predictions = np.full((umap2d_result.shape[0], 2), -1)  # Initialize with -1 for components

        for label in unique_labels:
            lowest_bic = np.inf
            best_gmm = None
            for n_components in range(1, max_components + 1):
                gmm = GaussianMixture(n_components=n_components, random_state=0)
                gmm.fit(umap2d_result[labels == label])
                bic = gmm.bic(umap2d_result[labels == label])
                if bic < lowest_bic:
                    lowest_bic = bic
                    best_gmm = gmm

            # Initial prediction for best GMM
            gmm_labels = best_gmm.predict(umap2d_result[labels == label])
            gmms[label] = best_gmm

            # Remove outliers: remove GMM components with small weight
            if ignore_gmm_weight_threshold is not None:
                valid_components = best_gmm.weights_ > ignore_gmm_weight_threshold
                valid_gmm_labels = np.where(valid_components[gmm_labels], gmm_labels, -1)

                filtered_gmm = GaussianMixture(n_components=sum(valid_components), random_state=0)
                filtered_gmm.weights_ = best_gmm.weights_[valid_components]
                filtered_gmm.means_ = best_gmm.means_[valid_components]
                filtered_gmm.covariances_ = best_gmm.covariances_[valid_components]
                filtered_gmm.precisions_ = best_gmm.precisions_[valid_components]
                filtered_gmm.precisions_cholesky_ = best_gmm.precisions_cholesky_[valid_components]

                filtered_gmms[label] = filtered_gmm

                # Update predictions with filtered GMM components
                for i, idx in enumerate(np.where(labels == label)[0]):
                    predictions[idx] = (label, valid_gmm_labels[i])
            else:
                for i, idx in enumerate(np.where(labels == label)[0]):
                    predictions[idx] = (label, gmm_labels[i])

        return gmms, filtered_gmms, predictions

    @staticmethod
    def _plot_n_sigma_ellipse(ax, mean, cov, weight, color, n_sigma=1, alpha=0.5):
        v, w = np.linalg.eigh(cov)
        v = n_sigma * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi
        ell = Ellipse(mean, v[0], v[1], 180.0 + angle, color=color, alpha=alpha, linewidth=5, fill=False)
        ax.add_patch(ell)

    def _save_gmm_imgs(self,
                       sorted_loces: Dict[int, Dict[int, LoCEMultilayerStorage]],
                       out_dir: str,
                       categories: Iterable[int],
                       dataset_tag: Literal["mscoco", "voc"],
                       image_tile_size: Tuple[int, int] = (256, 192),
                       frame_width: int = 5,
                       extra_tag: str = ""
                       ) -> None:
        
        imgs_dir = os.path.join(out_dir, f'loce_umap_gmm_imgs_tiles{extra_tag}')
        imgs_dir_outliers = os.path.join(out_dir, f'loce_umap_gmm_imgs_tiles_outliers{extra_tag}')
        rmdir(imgs_dir)
        mkdir(imgs_dir)
        rmdir(imgs_dir_outliers)
        mkdir(imgs_dir_outliers)

        if dataset_tag == "voc":
            selected_tag_id_names, selected_tag_ids_and_colors = draw_voc_categories_and_colors(categories)
        else:
            selected_tag_id_names, selected_tag_ids_and_colors = draw_mscoco_categories_and_colors(categories)

        for category_idx, gmm_content in sorted_loces.items():

            # number of images per gmm
            group_counts = {k: len(v) for k, v in gmm_content.items()}

            # evaluation of tile cells
            tile_cells = {k: find_closest_square_rootable_number(v) for k, v in group_counts.items()}

            for gmm_component_idx, gmm_loce_group in gmm_content.items():

                if gmm_component_idx == -1:
                    selected_imgs_dir = imgs_dir_outliers
                else:
                    selected_imgs_dir = imgs_dir

                tc = tile_cells[gmm_component_idx]

                # finalize number of rows and cols
                cols = rows = int(math.sqrt(tc))
                for r in range(rows, 0, -1):
                    if (r * cols) >= len(gmm_loce_group):
                        rows = r
                    else:
                        break

                # canvas
                canvas_width = image_tile_size[0] * cols
                canvas_height = image_tile_size[1] * rows
                canvas = Image.new("RGB", (canvas_width, canvas_height), color='white')

                # paste into canvas
                for row in range(rows):
                    for col in range(cols):
                        # offsets
                        x_offset = col * image_tile_size[0]
                        y_offset = row * image_tile_size[1]

                        # id
                        temp_idx = row * cols + col

                        # leave space empty if no more images
                        if temp_idx >= len(gmm_loce_group):
                            continue

                        temp_loce = gmm_loce_group[temp_idx]

                        # get current image, segmentation and color
                        temp_category = temp_loce.segmentation_category_id
                        temp_category_name = selected_tag_id_names[temp_category]
                        # fix mapping of model categories and MS COCO original categories
                        model_category = temp_category

                        temp_image_path = temp_loce.image_path
                        temp_seg = temp_loce.segmentation
                        # temp_selected_predictions = temp_loce.image_predictions
                        # temp_selected_predictions = [pred for pred in temp_loce.image_predictions if int(pred[5]) == model_category]
                        temp_color = selected_tag_ids_and_colors[temp_category]
                        temp_color_rgb = tuple([int(c * 255) for c in colors.to_rgb(temp_color)])
                        # blend image end segmentation (add colored mask)
                        colored_mask = get_colored_mask_alt(temp_seg, color_channels=[0, 1, 2], mask_value_multipliers=temp_color_rgb)

                        img_with_seg = blend_imgs(colored_mask, Image.open(temp_image_path), alpha=0.66)
                            
                        img_with_seg_counturs = Image.fromarray(add_countours_around_mask(np.array(img_with_seg), temp_seg, countour_color=temp_color_rgb, thickness=5))
                        img_with_seg_resized = img_with_seg_counturs.resize(image_tile_size)

                        # add colored frame
                        draw_frame(ImageDraw.Draw(img_with_seg_resized), image_tile_size, temp_color, frame_width)

                        canvas.paste(img_with_seg_resized, (x_offset, y_offset))

                canvas.save(f'{selected_imgs_dir}/category_{category_idx}_gmm_{gmm_component_idx+1}.jpg')

    def _umap_2d_plot_gmm_density_one_plot(self,
                                           group_size: int,
                                           loce_storages: Dict[int, Iterable[LoCEMultilayerStorage]],
                                           categories: Iterable[int],
                                           out_dir: str,
                                           distance: str,
                                           net_tag: str,
                                           layer: str,
                                           category_set_name: str,
                                           plot_2d_kwargs,
                                           save_gmm_imgs: bool,
                                           dataset_tag: Literal["mscoco", "voc"],):
        
        if dataset_tag == "voc":
            selected_tag_id_names, selected_tag_ids_and_colors = draw_voc_categories_and_colors(categories)
            selected_markers = VOC_MARKERS
        else:
            selected_tag_id_names, selected_tag_ids_and_colors = draw_mscoco_categories_and_colors(categories)
            selected_markers = MSCOCO_MARKERS

        loces, labels, clusters_flat = self._get_info_from_storages(loce_storages, layer)

        # umap fit
        umap2d = UMAP(n_neighbors=group_size, n_components=2, random_state=self.seed, metric=distance)
        umap2d_result = umap2d.fit_transform(loces)
        unique_labels = np.unique(labels)

        colors = [selected_tag_ids_and_colors[label] for label in labels]
        markers = [selected_markers[label] for label in labels]
        mesh_cmap = ListedColormap([selected_tag_ids_and_colors[label] for label in unique_labels])

        # gmms fit
        all_gmms, filtered_gmms, predictions = self._fit_gmms_without_outliers(umap2d_result, labels)
        gmms = filtered_gmms

        # sorted loces according to GMMs
        sorted_loces = {label: {} for label in unique_labels}
        for idx, (label, gmm_component_idx) in enumerate(predictions):
            if gmm_component_idx not in sorted_loces[label]:
                sorted_loces[label][gmm_component_idx] = []
            sorted_loces[label][gmm_component_idx].append(clusters_flat[idx])

        if save_gmm_imgs:
            self._save_gmm_imgs(sorted_loces, out_dir, categories, dataset_tag)

        # mesh eval 
        x = np.linspace(umap2d_result[:, 0].min() - 1, umap2d_result[:, 0].max() + 1, 200)
        y = np.linspace(umap2d_result[:, 1].min() - 1, umap2d_result[:, 1].max() + 1, 200)
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
        
        label_mesh_values = dict()

        for label in unique_labels:
            gmm = gmms[label]
            # countours
            Z = gmm.score_samples(XX)
            Z = Z.reshape(X.shape)
            label_mesh_values[label] = Z

        label_mesh_3d = np.stack([label_mesh_values[l] for l in unique_labels])
        max_idx_mesh = np.argmax(label_mesh_3d, axis=0)
        labels_mesh = unique_labels[max_idx_mesh]

        # plotting
        legend_width = 1.75
        text_effect = withStroke(linewidth=2, foreground='white')
        fig, axes = plt.subplots(1, len(plot_2d_kwargs), figsize=(6 * len(plot_2d_kwargs) + legend_width, 6), sharex=True, sharey=True)
        
        for col, (setup_name, (alpha_points, alpha_gmm, alpha_mesh)) in enumerate(plot_2d_kwargs.items()):

            if len(plot_2d_kwargs) == 1:
                ax = axes
            else:
                ax = axes[col]

            if alpha_mesh > 0:
                for label in unique_labels:
                    mask = (labels_mesh == label)
                    ax.contourf(X, Y, mask, levels=[0.5, 1], colors=[selected_tag_ids_and_colors[label]], alpha=alpha_mesh)

            if alpha_points > 0:
                for i in range(len(umap2d_result)):
                    ax.scatter(umap2d_result[i, 0], umap2d_result[i, 1], c=colors[i], marker=markers[i], alpha=alpha_points, s=self.marker_size)

            if alpha_gmm > 0:
                for label, gmm in gmms.items():
                    for i in range(gmm.n_components):
                        current_color = selected_tag_ids_and_colors[label]
                        self._plot_n_sigma_ellipse(ax, gmm.means_[i], gmm.covariances_[i], gmm.weights_[i], color=current_color, alpha=alpha_gmm)
                        ax.text(gmm.means_[i, 0], gmm.means_[i, 1], str(i+1), color=current_color, fontsize=10, ha='center', va='center',
                                fontweight='bold', path_effects=[text_effect])

            ax.set_xlabel('UMAP Component 1', fontsize=20)
            if col == 0:
                ax.set_ylabel('UMAP Component 2', fontsize=20)

            ax.set_xlim([umap2d_result[:, 0].min() - 0.2, umap2d_result[:, 0].max() + 0.2])
            ax.set_ylim([umap2d_result[:, 1].min() - 0.2, umap2d_result[:, 1].max() + 0.2])

        # Add one legend on the right side
        handles = [plt.Line2D([], [], color=selected_tag_ids_and_colors[label], marker=selected_markers[label], linestyle='', markersize=10, label=selected_tag_id_names[label]) for label in unique_labels]
        
        # Position the legend outside the plot area
        fig.legend(handles=handles, loc='center right', fontsize=14)

        fig.suptitle(f"{LoCEExperimentConstants.NET_FULL_NAMES[net_tag]}: {layer}", fontsize=20)

        plot_width_part = (6 * len(plot_2d_kwargs)) / (6 * len(plot_2d_kwargs) + legend_width)
        plt.tight_layout(rect=[0, 0, plot_width_part, 1])
        plt.savefig(os.path.join(out_dir, f"umap_{net_tag}_{layer}_{category_set_name}_{distance}_{len(plot_2d_kwargs)}.pdf"))
        plt.close()

    def _umap_2d_plot_gmm_density_one_plot_ignore_labels(self,
                                                         group_size: int,
                                                         loce_storages: Dict[int, Iterable[LoCEMultilayerStorage]],
                                                         categories: Iterable[int],
                                                         out_dir: str,
                                                         distance: str,
                                                         net_tag: str,
                                                         layer: str,
                                                         category_set_name: str,
                                                         plot_2d_kwargs,
                                                         save_gmm_imgs: bool,
                                                         dataset_tag: Literal["mscoco", "voc"],):
        
        def generate_grey_colors(n):
            return [(i/n, i/n, i/n) for i in range(1, n + 1)]

        if dataset_tag == "voc":
            selected_tag_id_names, selected_tag_ids_and_colors = draw_voc_categories_and_colors(categories)
            selected_markers = VOC_MARKERS
        else:
            selected_tag_id_names, selected_tag_ids_and_colors = draw_mscoco_categories_and_colors(categories)
            selected_markers = MSCOCO_MARKERS

        loces, labels, clusters_flat = self._get_info_from_storages(loce_storages, layer)

        unique_labels_legend = np.unique(labels)

        colors = [selected_tag_ids_and_colors[label] for label in labels]
        markers = [selected_markers[label] for label in labels]

        labels = np.full(labels.shape, -1)

        # umap fit
        umap2d = UMAP(n_neighbors=group_size, n_components=2, random_state=self.seed, metric=distance)
        umap2d_result = umap2d.fit_transform(loces)
        unique_labels = np.unique(labels)


        # gmms fit
        all_gmms, filtered_gmms, predictions = self._fit_gmms_without_outliers(umap2d_result, labels, max_components=10)
        gmms = all_gmms

        # sorted loces according to GMMs
        sorted_loces = {label: {} for label in unique_labels}

        for idx, (label, gmm_component_idx) in enumerate(predictions):
            if gmm_component_idx not in sorted_loces[label]:
                sorted_loces[label][gmm_component_idx] = []
            sorted_loces[label][gmm_component_idx].append(clusters_flat[idx])

        if save_gmm_imgs:
            self._save_gmm_imgs(sorted_loces, out_dir, categories, dataset_tag, extra_tag="_ignored_labels")

        # mesh eval 
        x = np.linspace(umap2d_result[:, 0].min() - 1, umap2d_result[:, 0].max() + 1, 200)
        y = np.linspace(umap2d_result[:, 1].min() - 1, umap2d_result[:, 1].max() + 1, 200)
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
        
        label_mesh_values = dict()

        # Get the GMM for label -1
        gmm = gmms[-1]

        # Evaluate the log-likelihood of each component
        log_prob = gmm._estimate_log_prob(XX)

        # Reshape the log-likelihoods to match the mesh grid
        log_prob_mesh = log_prob.reshape((XX.shape[0], gmm.n_components)).T
        log_prob_mesh = log_prob_mesh.reshape((gmm.n_components, X.shape[0], X.shape[1]))

        # Find the component with the highest likelihood for each point in the mesh
        max_idx_mesh = np.argmax(log_prob_mesh, axis=0)

        labels_mesh = max_idx_mesh

        gmm_unique_labels = np.unique(max_idx_mesh)

        #colors_gmms = plt.cm.tab10.colors 
        colors_gmms = generate_grey_colors(10)
        color_mapping_gmm = {label: colors_gmms[i % len(colors_gmms)] for i, label in enumerate(gmm_unique_labels)}


        # plotting
        legend_width = 1.75
        text_effect = withStroke(linewidth=2, foreground='white')
        fig, axes = plt.subplots(1, len(plot_2d_kwargs), figsize=(6 * len(plot_2d_kwargs) + legend_width, 6), sharex=True, sharey=True)
        
        for col, (setup_name, (alpha_points, alpha_gmm, alpha_mesh)) in enumerate(plot_2d_kwargs.items()):

            if len(plot_2d_kwargs) == 1:
                ax = axes
            else:
                ax = axes[col]

            if alpha_mesh > 0:
                for label in gmm_unique_labels:
                    mask = (labels_mesh == label)
                    ax.contourf(X, Y, mask, levels=[0.5, 1], colors=[color_mapping_gmm[label]], alpha=alpha_mesh)

            if alpha_points > 0:
                for i in range(len(umap2d_result)):
                    ax.scatter(umap2d_result[i, 0], umap2d_result[i, 1], c=colors[i], marker=markers[i], alpha=alpha_points, s=self.marker_size)

            if alpha_gmm > 0:
                for label, gmm in gmms.items():
                    for i in range(gmm.n_components):
                        current_color = color_mapping_gmm[i]
                        self._plot_n_sigma_ellipse(ax, gmm.means_[i], gmm.covariances_[i], gmm.weights_[i], color=current_color, alpha=alpha_gmm)
                        ax.text(gmm.means_[i, 0], gmm.means_[i, 1], str(i+1), color=current_color, fontsize=10, ha='center', va='center',
                                fontweight='bold', path_effects=[text_effect])


            ax.set_xlabel('UMAP Component 1', fontsize=20)
            if col == 0:
                ax.set_ylabel('UMAP Component 2', fontsize=20)

            ax.set_xlim([umap2d_result[:, 0].min() - 0.2, umap2d_result[:, 0].max() + 0.2])
            ax.set_ylim([umap2d_result[:, 1].min() - 0.2, umap2d_result[:, 1].max() + 0.2])

        # Add one legend on the right side
        handles = [plt.Line2D([], [], color=selected_tag_ids_and_colors[label], marker=selected_markers[label], linestyle='', markersize=10, label=selected_tag_id_names[label]) for label in unique_labels_legend]
        
        # Position the legend outside the plot area
        fig.legend(handles=handles, loc='center right', fontsize=14)

        fig.suptitle(f"{LoCEExperimentConstants.NET_FULL_NAMES[net_tag]}: {layer}", fontsize=20)

        plot_width_part = (6 * len(plot_2d_kwargs)) / (6 * len(plot_2d_kwargs) + legend_width)
        plt.tight_layout(rect=[0, 0, plot_width_part, 1])
        plt.savefig(os.path.join(out_dir, f"umap_{net_tag}_{layer}_{category_set_name}_{distance}_{len(plot_2d_kwargs)}_ignored_labels.pdf"))
        plt.close()

    def evaluate_many(self,
                      net_tags: Iterable[str] = LoCEExperimentConstants.NET_TAGS,
                      layers_dict: Dict[str, Iterable[str]] = LoCEExperimentConstants.IOU_LAYERS,
                      plot_hierarchial_clustering_dendrogram: bool = True,
                      save_hierarchial_clustering_imgs: bool = False,
                      save_gmm_imgs: bool = False,
                      gmm_ignores_labels: bool = False,
                      dataset_tag: Literal["mscoco", "voc"] = "mscoco"
                      ) -> None:
        for net_tag in net_tags:
    
            if dataset_tag == "voc":
                res_dir = LoCEExperimentConstants.LoCE_DIR_VOC[net_tag]
            else:
                res_dir = LoCEExperimentConstants.LoCE_DIR[net_tag]

            loce_loader = LoCEMultilayerStorageDirectoryLoader(res_dir, seed=self.seed, min_seg_area=0, max_seg_area=1)
            
            layers = layers_dict[net_tag]

            self.evaluate_one(loce_loader, net_tag, layers, plot_hierarchial_clustering_dendrogram, save_hierarchial_clustering_imgs, save_gmm_imgs, gmm_ignores_labels, dataset_tag)

    def evaluate_one(self,
                     loce_loader: LoCEMultilayerStorageDirectoryLoader,
                     net_tag: str,
                     layers: Iterable[str],
                     plot_hierarchial_clustering_dendrogram: bool = True,
                     save_hierarchial_clustering_imgs: bool = False,
                     save_gmm_imgs: bool = False,
                     gmm_ignores_labels: bool = False,
                     dataset_tag: Literal["mscoco", "voc"] = "mscoco"
                     ) -> None:
        
        if dataset_tag == "voc":
            category_sets = self.category_sets_voc
        else:
            category_sets = self.category_sets

        for layer in layers:

            for category_set_name, categories in category_sets.items():

                loce_storages = loce_loader.load(categories)

                group_size = int(sum([len(l) for l in loce_storages.values()]) * self.sample_threshold_coefficient)

                for method, distance in self.method_distance:

                    print(f"\n{LoCEExperimentConstants.NET_FULL_NAMES[net_tag]}.{layer}.{category_set_name}.{distance}")

                    out_dir = os.path.join(self.out_dir_base, f"{net_tag}/{category_set_name}/{layer}_{method}_{distance}")
                    mkdir(out_dir)

                    #######################################################################

                    if plot_hierarchial_clustering_dendrogram:
                        if dataset_tag == "voc":
                            selected_tag_id_names, selected_tag_ids_and_colors = draw_voc_categories_and_colors(categories)
                        else:
                            selected_tag_id_names, selected_tag_ids_and_colors = draw_mscoco_categories_and_colors(categories)
                        loce_clusterer = LoCEClusterer(selected_tag_ids_and_colors, selected_tag_id_names, [layer], self.n_samples_per_tag, group_size, self.cluster_purity, distance, method, out_dir)
                        clusters = loce_clusterer.cluster_loces(loce_storages, save_proj=False, save_imgs=save_hierarchial_clustering_imgs)
                        loce_clusterer.loce_dendogram(loce_storages, dataset_tag=dataset_tag)

                    if gmm_ignores_labels:
                        self._umap_2d_plot_gmm_density_one_plot_ignore_labels(group_size, loce_storages, categories, out_dir, distance, net_tag, layer, category_set_name, self.alpha_kwargs_grey_current, save_gmm_imgs, dataset_tag)
                    else:
                        self._umap_2d_plot_gmm_density_one_plot(group_size, loce_storages, categories, out_dir, distance, net_tag, layer, category_set_name, self.alpha_kwargs_current, save_gmm_imgs, dataset_tag)

    def evaluate_many_single_plot(self,
                                  loce_loaders: Dict[str, LoCEMultilayerStorageDirectoryLoader],
                                  layers_dict: Dict[str, Iterable[str]] = LoCEExperimentConstants.IOU_LAYERS,
                                  dataset_tag: Literal["mscoco", "voc"] = "mscoco",
                                  hac_clustering_type: Literal['adaptive', 'linkage', 'size'] = 'adaptive',
                                  ) -> None:
        for net_tag, loce_loader in loce_loaders.items():
                        
            layers = layers_dict[net_tag]

            self.evaluate_one_single_plot(loce_loader, net_tag, layers, dataset_tag, hac_clustering_type)

    def evaluate_one_single_plot(self,
                                 loce_loader: LoCEMultilayerStorageDirectoryLoader,
                                 net_tag: str,
                                 layers: Iterable[str],
                                 dataset_tag: Literal["mscoco", "voc"] = "mscoco",
                                 hac_clustering_type: Literal['adaptive', 'linkage', 'size'] = 'adaptive',
                                 hac_clustering_kwargs: Dict[str, Any] = {}
                                 ) -> None:

        if dataset_tag == "voc":
            category_sets = self.category_sets_voc
        else:
            category_sets = self.category_sets

        for layer in layers:

            for category_set_name, categories in category_sets.items():

                loce_storages = loce_loader.load(categories)

                group_size = int(sum([len(l) for l in loce_storages.values()]) * self.sample_threshold_coefficient)

                for method, distance in self.method_distance:

                    print(f"\n{LoCEExperimentConstants.NET_FULL_NAMES[net_tag]}.{layer}.{category_set_name}.{distance}")

                    out_dir = os.path.join(self.out_dir_base, f"{net_tag}/{category_set_name}/{layer}_{method}_{distance}")
                    mkdir(out_dir)

                    #######################################################################

                    base_alpha = 0.9

                    fig_width = 20
                    fig_height = 9

                    margin = 0.025  # This is now used as the margin between plots as well

                    plt.figure(figsize=(fig_width, fig_height))

                    # Calculate dimensions for the bottom plot (aspect ratio 0.1)
                    bottom_width = 1 - 2 * margin
                    bottom_height = bottom_width * 0.1  # height/width = 0.1
                    bottom_height_norm = bottom_height * fig_width / fig_height  # Normalize to figure height

                    # Calculate remaining height for top square plots
                    remaining_height = 1 - bottom_height_norm - 3 * margin  # Top margin, bottom margin, and space above bottom plot
                    square_width = min(remaining_height, (1 - 4 * margin) / 3)  # Ensure squares fit width-wise with margins
                    square_height = square_width * fig_width / fig_height

                    # bottom
                    ax1 = plt.subplot(212)
                    ax1.set_position([margin, margin + 0.05, bottom_width, bottom_height_norm - 0.05 + margin])

                    # top left
                    ax2 = plt.subplot(231)
                    ax2.set_position([margin, 1 - square_height - margin, square_width, square_height])

                    # top middle
                    ax3 = plt.subplot(232)
                    ax3.set_position([0.5 - square_width/2, 1 - square_height - margin, square_width, square_height])

                    # top right
                    ax4 = plt.subplot(233)
                    ax4.set_position([1 - margin - square_width, 1 - square_height - margin, square_width, square_height])

                    # dendrogram
                    if dataset_tag == "voc":
                        selected_tag_id_names, selected_tag_ids_and_colors = draw_voc_categories_and_colors(categories)
                    else:
                        selected_tag_id_names, selected_tag_ids_and_colors = draw_mscoco_categories_and_colors(categories)

                    loce_clusterer = LoCEClusterer(selected_tag_ids_and_colors, selected_tag_id_names, [layer], self.n_samples_per_tag, group_size, self.cluster_purity, distance, method, out_dir, **hac_clustering_kwargs)
                    # clusters = loce_clusterer.cluster_loces(loce_storages, save_proj=False, save_imgs=save_hierarchial_clustering_imgs)
                    loce_clusterer.loce_dendogram(loce_storages, ax=ax1, dataset_tag=dataset_tag, clustering_type=hac_clustering_type)

                    # dots
                    alphas1 = (base_alpha, 0, 0)
                    self._umap_2d_plot_gmm_density_one_plot2(group_size, loce_storages, categories, out_dir, distance, net_tag, layer, category_set_name, False, alphas1, dataset_tag, ax=ax2)

                    # gmms
                    alphas2 = (base_alpha/2, base_alpha, base_alpha/4)
                    self._umap_2d_plot_gmm_density_one_plot2(group_size, loce_storages, categories, out_dir, distance, net_tag, layer, category_set_name, True, alphas2, dataset_tag, ax=ax3)

                    # gmms ignore label                    
                    alphas3 = (base_alpha/2, base_alpha, base_alpha/2)
                    self._umap_2d_plot_gmm_density_one_plot_ignore_labels2(group_size, loce_storages, categories, out_dir, distance, net_tag, layer, category_set_name, True, alphas3, dataset_tag, ax=ax4)

                    #plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f"umap_{net_tag}_{layer}_{category_set_name}_{distance}_one_plot.pdf"))
                    plt.savefig(os.path.join(out_dir, f"umap_{net_tag}_{layer}_{category_set_name}_{distance}_one_plot.jpg"))
                    plt.close()
                    
    def _umap_2d_plot_gmm_density_one_plot2(self,
                                            group_size: int,
                                            loce_storages: Dict[int, Iterable[LoCEMultilayerStorage]],
                                            categories: Iterable[int],
                                            out_dir: str,
                                            distance: str,
                                            net_tag: str,
                                            layer: str,
                                            category_set_name: str,
                                            save_gmm_imgs: bool,
                                            alphas: Tuple[float, float, float],
                                            dataset_tag: Literal["mscoco", "voc"],
                                            ax: plt.Axes = None):
        
        if dataset_tag == "voc":
            selected_tag_id_names, selected_tag_ids_and_colors = draw_voc_categories_and_colors(categories)
            selected_markers = VOC_MARKERS
        else:
            selected_tag_id_names, selected_tag_ids_and_colors = draw_mscoco_categories_and_colors(categories)
            selected_markers = MSCOCO_MARKERS

        loces, labels, clusters_flat = self._get_info_from_storages(loce_storages, layer)

        # umap fit
        umap2d = UMAP(n_neighbors=group_size, n_components=2, random_state=self.seed, metric=distance)
        umap2d_result = umap2d.fit_transform(loces)
        unique_labels = np.unique(labels)

        colors = [selected_tag_ids_and_colors[label] for label in labels]
        markers = [selected_markers[label] for label in labels]
        mesh_cmap = ListedColormap([selected_tag_ids_and_colors[label] for label in unique_labels])

        # gmms fit
        all_gmms, filtered_gmms, predictions = self._fit_gmms_without_outliers(umap2d_result, labels)
        gmms = filtered_gmms

        # sorted loces according to GMMs
        sorted_loces = {label: {} for label in unique_labels}
        for idx, (label, gmm_component_idx) in enumerate(predictions):
            if gmm_component_idx not in sorted_loces[label]:
                sorted_loces[label][gmm_component_idx] = []
            sorted_loces[label][gmm_component_idx].append(clusters_flat[idx])

        if save_gmm_imgs:
            self._save_gmm_imgs(sorted_loces, out_dir, categories, dataset_tag)

        # mesh eval 
        x = np.linspace(umap2d_result[:, 0].min() - 1, umap2d_result[:, 0].max() + 1, 200)
        y = np.linspace(umap2d_result[:, 1].min() - 1, umap2d_result[:, 1].max() + 1, 200)
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
        
        label_mesh_values = dict()

        for label in unique_labels:
            gmm = gmms[label]
            # countours
            Z = gmm.score_samples(XX)
            Z = Z.reshape(X.shape)
            label_mesh_values[label] = Z

        label_mesh_3d = np.stack([label_mesh_values[l] for l in unique_labels])
        max_idx_mesh = np.argmax(label_mesh_3d, axis=0)
        labels_mesh = unique_labels[max_idx_mesh]

        # plotting
        legend_width = 1.75
        text_effect = withStroke(linewidth=2, foreground='white')

        alpha_points, alpha_gmm, alpha_mesh = alphas
        
        if alpha_mesh > 0:
            for label in unique_labels:
                mask = (labels_mesh == label)
                ax.contourf(X, Y, mask, levels=[0.5, 1], colors=[selected_tag_ids_and_colors[label]], alpha=alpha_mesh)

        if alpha_points > 0:
            for i in range(len(umap2d_result)):
                ax.scatter(umap2d_result[i, 0], umap2d_result[i, 1], c=colors[i], marker=markers[i], alpha=alpha_points, s=self.marker_size)

        if alpha_gmm > 0:
            for label, gmm in gmms.items():
                for i in range(gmm.n_components):
                    current_color = selected_tag_ids_and_colors[label]
                    self._plot_n_sigma_ellipse(ax, gmm.means_[i], gmm.covariances_[i], gmm.weights_[i], color=current_color, alpha=alpha_gmm)
                    ax.text(gmm.means_[i, 0], gmm.means_[i, 1], str(i+1), color=current_color, fontsize=13, ha='center', va='center',
                            fontweight='bold', path_effects=[text_effect])

        ax.set_xlabel('UMAP Component 1', fontsize=20)
        ax.set_ylabel('UMAP Component 2', fontsize=20)

        ax.set_xlim([umap2d_result[:, 0].min() - 0.2, umap2d_result[:, 0].max() + 0.2])
        ax.set_ylim([umap2d_result[:, 1].min() - 0.2, umap2d_result[:, 1].max() + 0.2])

        ax.tick_params(axis='both', which='both', bottom=False, top=False,
               left=False, right=False, labelbottom=False, labelleft=False)

        # Add one legend on the right side
        handles = [plt.Line2D([], [], color=selected_tag_ids_and_colors[label], marker=selected_markers[label], linestyle='', markersize=10, label=selected_tag_id_names[label]) for label in unique_labels]
        
        # Position the legend outside the plot area
        #ax.legend(handles=handles, loc='center right', fontsize=14)

        #ax.set_title(f"{LoCEExperimentConstants.NET_FULL_NAMES[net_tag]}: {layer}", fontsize=15)

    def _umap_2d_plot_gmm_density_one_plot_ignore_labels2(self,
                                                          group_size: int,
                                                          loce_storages: Dict[int, Iterable[LoCEMultilayerStorage]],
                                                          categories: Iterable[int],
                                                          out_dir: str,
                                                          distance: str,
                                                          net_tag: str,
                                                          layer: str,
                                                          category_set_name: str,
                                                          save_gmm_imgs: bool,
                                                          alphas: Tuple[float, float, float],
                                                          dataset_tag: Literal["mscoco", "voc"],
                                                          ax: plt.Axes = None):
        
        def generate_grey_colors(n):
            return [(i/n, i/n, i/n) for i in range(1, n + 1)]

        if dataset_tag == "voc":
            selected_tag_id_names, selected_tag_ids_and_colors = draw_voc_categories_and_colors(categories)
            selected_markers = VOC_MARKERS
        else:
            selected_tag_id_names, selected_tag_ids_and_colors = draw_mscoco_categories_and_colors(categories)
            selected_markers = MSCOCO_MARKERS

        loces, labels, clusters_flat = self._get_info_from_storages(loce_storages, layer)

        unique_labels_legend = np.unique(labels)

        colors = [selected_tag_ids_and_colors[label] for label in labels]
        markers = [selected_markers[label] for label in labels]

        labels = np.full(labels.shape, -1)

        # umap fit
        umap2d = UMAP(n_neighbors=group_size, n_components=2, random_state=self.seed, metric=distance)
        umap2d_result = umap2d.fit_transform(loces)
        unique_labels = np.unique(labels)

        max_gmm_components = len(categories) * 2

        # gmms fit
        all_gmms, filtered_gmms, predictions = self._fit_gmms_without_outliers(umap2d_result, labels, max_components=max_gmm_components)
        gmms = all_gmms

        # sorted loces according to GMMs
        sorted_loces = {label: {} for label in unique_labels}

        for idx, (label, gmm_component_idx) in enumerate(predictions):
            if gmm_component_idx not in sorted_loces[label]:
                sorted_loces[label][gmm_component_idx] = []
            sorted_loces[label][gmm_component_idx].append(clusters_flat[idx])

        if save_gmm_imgs:
            self._save_gmm_imgs(sorted_loces, out_dir, categories, dataset_tag, extra_tag="_ignored_labels")

        # mesh eval 
        x = np.linspace(umap2d_result[:, 0].min() - 1, umap2d_result[:, 0].max() + 1, 200)
        y = np.linspace(umap2d_result[:, 1].min() - 1, umap2d_result[:, 1].max() + 1, 200)
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
        
        label_mesh_values = dict()

        # Get the GMM for label -1
        gmm = gmms[-1]
        total_gmm_components = gmm.n_components

        # Evaluate the log-likelihood of each component
        log_prob = gmm._estimate_log_prob(XX)

        # Reshape the log-likelihoods to match the mesh grid
        log_prob_mesh = log_prob.reshape((XX.shape[0], gmm.n_components)).T
        log_prob_mesh = log_prob_mesh.reshape((gmm.n_components, X.shape[0], X.shape[1]))

        # Find the component with the highest likelihood for each point in the mesh
        max_idx_mesh = np.argmax(log_prob_mesh, axis=0)

        labels_mesh = max_idx_mesh

        gmm_unique_labels = np.unique(max_idx_mesh)

        #colors_gmms = plt.cm.tab10.colors
        colors_gmms = generate_grey_colors(total_gmm_components + 3)  # + 3 to avoid absolutely white 
        color_mapping_gmm = {label: colors_gmms[i % len(colors_gmms)] for i, label in enumerate(gmm_unique_labels)}

        # plotting
        legend_width = 1.75
        text_effect = withStroke(linewidth=2, foreground='white')
        
        alpha_points, alpha_gmm, alpha_mesh = alphas

        if alpha_mesh > 0:
            for label in gmm_unique_labels:
                mask = (labels_mesh == label)
                ax.contourf(X, Y, mask, levels=[0.5, 1], colors=[color_mapping_gmm[label]], alpha=alpha_mesh)
                ax.contour(X, Y, mask, levels=[0.5, 1], colors='black', linewidths=1, linestyles="dashed")

        if alpha_points > 0:
            for i in range(len(umap2d_result)):
                ax.scatter(umap2d_result[i, 0], umap2d_result[i, 1], c=colors[i], marker=markers[i], alpha=alpha_points, s=self.marker_size)

        if alpha_gmm > 0:
            for label, gmm in gmms.items():
                for i in range(gmm.n_components):
                    current_color = color_mapping_gmm[i]
                    self._plot_n_sigma_ellipse(ax, gmm.means_[i], gmm.covariances_[i], gmm.weights_[i], color=current_color, alpha=alpha_gmm)
                    ax.text(gmm.means_[i, 0], gmm.means_[i, 1], str(i+1), color=current_color, fontsize=13, ha='center', va='center',
                            fontweight='bold', path_effects=[text_effect])


        ax.set_xlabel('UMAP Component 1', fontsize=20)
        ax.set_ylabel('UMAP Component 2', fontsize=20)

        ax.set_xlim([umap2d_result[:, 0].min() - 0.2, umap2d_result[:, 0].max() + 0.2])
        ax.set_ylim([umap2d_result[:, 1].min() - 0.2, umap2d_result[:, 1].max() + 0.2])

        ax.tick_params(axis='both', which='both', bottom=False, top=False,
               left=False, right=False, labelbottom=False, labelleft=False)

        # Add one legend on the right side
        handles = [plt.Line2D([], [], color=selected_tag_ids_and_colors[label], marker=selected_markers[label], linestyle='', markersize=10, label=selected_tag_id_names[label]) for label in unique_labels_legend]
        
        # Position the legend outside the plot area
        #ax.legend(handles=handles, loc='center right', fontsize=14)

        #ax.set_title(f"{LoCEExperimentConstants.NET_FULL_NAMES[net_tag]}: {layer}", fontsize=15)

    def evaluate_many_purities(self,
                               net_tags: Iterable[str] = LoCEExperimentConstants.NET_TAGS,
                               layers_dict: Dict[str, Iterable[str]] = LoCEExperimentConstants.IOU_LAYERS,
                               n_eval: int = 50,
                               dataset_tag: Literal["mscoco", "voc"] = "mscoco"
                               ) -> None:
        
        if dataset_tag == "voc":
            category_sets_purity = self.category_sets_purity_voc
        else:
            category_sets_purity = self.category_sets_purity
        
        mkdir(self.out_dir_base)

        res = {(method, distance): {category_set_name: {net_tag: {layer: [] for layer in layers_dict[net_tag]} for net_tag in net_tags} for category_set_name in category_sets_purity.keys()} for method, distance in self.method_distance}

        for seed in tqdm(range(n_eval)):

            for net_tag in net_tags:
        
                if dataset_tag == "voc":
                    res_dir = LoCEExperimentConstants.LoCE_DIR_VOC[net_tag]
                else:
                    res_dir = LoCEExperimentConstants.LoCE_DIR[net_tag]

                loce_loader = LoCEMultilayerStorageDirectoryLoader(res_dir, seed=seed, min_seg_area=0, max_seg_area=1)
                
                layers = layers_dict[net_tag]

                for layer in layers:

                    for category_set_name, categories in category_sets_purity.items():

                        loce_storages = loce_loader.load(categories)

                        group_size = int(sum([len(l) for l in loce_storages.values()]) * self.sample_threshold_coefficient)

                        for method, distance in self.method_distance:

                            #print(f"\n{LoCEExperimentConstants.NET_FULL_NAMES[net_tag]}.{layer}.{category_set_name}.{distance}")

                            out_dir = os.path.join(self.out_dir_base, f"{net_tag}/{category_set_name}/{layer}_{method}_{distance}")
                            mkdir(out_dir)

                            #######################################################################

                            if dataset_tag == "voc":
                                selected_tag_id_names, selected_tag_ids_and_colors = draw_voc_categories_and_colors(categories)
                            else:
                                selected_tag_id_names, selected_tag_ids_and_colors = draw_mscoco_categories_and_colors(categories)

                            loce_clusterer = LoCEClusterer(selected_tag_ids_and_colors, selected_tag_id_names, [layer], self.n_samples_per_tag, group_size, self.cluster_purity, distance, method, out_dir)

                            purity, n_clusters = loce_clusterer.get_clustering_state(loce_storages)

                            res[(method, distance)][category_set_name][net_tag][layer].append((purity, n_clusters))

        write_pickle(res, os.path.join(self.out_dir_base, "cluster_state.pkl")) 

    def evaluate_many_one_plot_confusion(self,
                                         net_tags: Iterable[str] = LoCEExperimentConstants.NET_TAGS,
                                         layers_dict: Dict[str, Iterable[str]] = LoCEExperimentConstants.IOU_LAYERS,
                                         cluster_idxs: Dict[str, Iterable[int]] = None,
                                         dataset_tag: Literal["mscoco", "voc"] = "mscoco"
                                         ) -> None:
        for net_tag in net_tags:
    
            if dataset_tag == "voc":
                res_dir = LoCEExperimentConstants.LoCE_DIR_VOC[net_tag]
            else:
                res_dir = LoCEExperimentConstants.LoCE_DIR[net_tag]

            loce_loader = LoCEMultilayerStorageDirectoryLoader(res_dir, seed=self.seed, min_seg_area=0, max_seg_area=1)
            
            layers = layers_dict[net_tag]

            if cluster_idxs is None:
                self.evaluate_one_one_plot_confusion(loce_loader, net_tag, layers, None, dataset_tag)
            else:
                self.evaluate_one_one_plot_confusion(loce_loader, net_tag, layers, cluster_idxs[net_tag], dataset_tag)
                

    def evaluate_one_one_plot_confusion(self,
                                        loce_loader: LoCEMultilayerStorageDirectoryLoader,
                                        net_tag: str,
                                        layers: Iterable[str],
                                        cluster_idxs: Iterable[int] = None,
                                        dataset_tag: Literal["mscoco", "voc"] = "mscoco"
                                        ) -> None:
        
        if dataset_tag == "voc":
            category_sets = self.category_sets_voc
        else:
            category_sets = self.category_sets

        for idx, layer in enumerate(layers):

            for category_set_name, categories in category_sets.items():

                loce_storages = loce_loader.load(categories)

                group_size = int(sum([len(l) for l in loce_storages.values()]) * self.sample_threshold_coefficient)

                for method, distance in self.method_distance:

                    print(f"\n{LoCEExperimentConstants.NET_FULL_NAMES[net_tag]}.{layer}.{category_set_name}.{distance}")

                    out_dir = os.path.join(self.out_dir_base, f"{net_tag}/{category_set_name}/{layer}_{method}_{distance}")
                    mkdir(out_dir)

                    #######################################################################

                    base_alpha = 0.9

                    fig_width = 20
                    fig_height = 9

                    margin = 0.025  # This is now used as the margin between plots as well

                    plt.figure(figsize=(fig_width, fig_height))

                    # Calculate dimensions for the bottom plot (aspect ratio 0.1)
                    bottom_width = 1 - 2 * margin
                    bottom_height = bottom_width * 0.1  # height/width = 0.1
                    bottom_height_norm = bottom_height * fig_width / fig_height  # Normalize to figure height

                    # Calculate remaining height for top square plots
                    remaining_height = 1 - bottom_height_norm - 3 * margin  # Top margin, bottom margin, and space above bottom plot
                    square_width = min(remaining_height, (1 - 4 * margin) / 3)  # Ensure squares fit width-wise with margins
                    square_height = square_width * fig_width / fig_height

                    # bottom
                    ax1 = plt.subplot(212)
                    ax1.set_position([margin, margin + 0.05, bottom_width, bottom_height_norm - 0.05 + margin])

                    # top left
                    ax2 = plt.subplot(231)
                    ax2.set_position([margin, 1 - square_height - margin, square_width, square_height])

                    # top middle
                    ax3 = plt.subplot(232)
                    ax3.set_position([0.5 - square_width/2, 1 - square_height - margin, square_width, square_height])

                    # top right
                    ax4 = plt.subplot(233)
                    ax4.set_position([1 - margin - square_width, 1 - square_height - margin, square_width, square_height])

                    # dendrogram
                    if dataset_tag == "voc":
                        selected_tag_id_names, selected_tag_ids_and_colors = draw_voc_categories_and_colors(categories)
                    else:
                        selected_tag_id_names, selected_tag_ids_and_colors = draw_mscoco_categories_and_colors(categories)

                    loce_clusterer = LoCEClusterer(selected_tag_ids_and_colors, selected_tag_id_names, [layer], self.n_samples_per_tag, group_size, self.cluster_purity, distance, method, out_dir)
                    # clusters = loce_clusterer.cluster_loces(loce_storages, save_proj=False, save_imgs=save_hierarchial_clustering_imgs)
                    loce_clusterer.loce_dendogram(loce_storages, ax=ax1, dataset_tag=dataset_tag)

                    if cluster_idxs is None:
                        # dots
                        alphas1 = (base_alpha, 0, 0)
                        self._umap_2d_plot_gmm_density_one_plot2(group_size, loce_storages, categories, out_dir, distance, net_tag, layer, category_set_name, True, alphas1, dataset_tag, ax=ax2)
                    else:
                        cluster_img = Image.open(os.path.join(out_dir, "loce_clustered_imgs_tiles", f"cluster_{cluster_idxs[idx]}.jpg"))
                        ax2.imshow(cluster_img)
                        ax2.set_title(f"Cluster {cluster_idxs[idx]}", fontsize=20)
                        ax2.axis('off')

                    # gmms
                    alphas2 = (base_alpha/2, base_alpha, base_alpha/4)
                    self._umap_2d_plot_gmm_density_one_plot2(group_size, loce_storages, categories, out_dir, distance, net_tag, layer, category_set_name, True, alphas2, dataset_tag, ax=ax3)

                    # gmms ignore label                    
                    alphas3 = (base_alpha/2, base_alpha, base_alpha/2)
                    self._umap_2d_plot_gmm_density_one_plot_ignore_labels2(group_size, loce_storages, categories, out_dir, distance, net_tag, layer, category_set_name, True, alphas3, dataset_tag, ax=ax4)

                    #plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f"umap_{net_tag}_{layer}_{category_set_name}_{distance}_one_plot.pdf"))
                    plt.close()