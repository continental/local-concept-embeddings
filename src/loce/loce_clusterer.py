'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

from xai_utils.logging import init_logger, log_info
init_logger()

from typing import Dict, List, Tuple, Iterable, Callable, Union, Literal
import os

from .loce_utils import LoCEActivationsTensorExtractor, ImageLoader, get_colored_mask, blend_imgs, combine_masks, get_projection, save_cluster_proj_as_tiles, save_cluster_imgs_as_tiles, get_rgb_binary_mask, MSCOCO_CATEGORY_COLORS, MSCOCO_MARKERS, VOC_MARKERS, VOC_CATEGORY_COLORS
from .loce import LoCEMultilayerStorage, LoCEMultilayerClusterInfo, LoCEMultilayerStorageDirectoryLoader
from xai_utils.files import mkdir, apply_heatmap, apply_mask, add_countours_around_mask, normalize_0_to_1, rmdir

from skimage.transform import resize
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster import hierarchy


class LoCEClusterer:

    def __init__(self,
                 selected_tag_ids_and_colors: Dict[int, str],
                 selected_tag_id_names: Dict[int, str],
                 loce_layers: List[str],
                 max_samples_per_category: int,
                 cluster_size_threshold: int = 20,
                 cluster_purity_threshold: float = 0.8,
                 distance_metric: str = 'cosine',
                 clustering_method: str = 'complete',
                 save_dir: str = './experiment_outputs/loce_clustering',
                 cluster_linkage_threshold: float = 4.0
                 ) -> None:
        """
        Args:
            selected_tag_ids_and_colors (Dict[int, str]): categories (tag_ids) to perform clustering for and corresponding colors, e.g., {3: 'red', 4: 'blue'}
            selected_tag_ids_and_colors (Dict[int, str]): categories (tag_ids) to perform clustering for and corresponding names, e.g., {3: 'car', 4: 'motorcycle'}
            loce_layers (List[str]): layers for LoCE concatenation (several layers may improve the result)
            max_samples_per_category (int): max number of samples per clustered category / concept, if category has less samples provided than this threshold - all samples will be used

        Kwargs:
            cluster_size_threshold (int = 20): 'adaptive' clustering: if size of the cluster is smaller than given value it will be saved and noth further decomposed into subclusters, 'size' clustering: threshold that defines max size of cluster
            cluster_purity_threshold (float = 0.8): if purity of cluster is higher than given value it will be saved and noth further decomposed into subclusters
            distance_metric (str = 'cosine'): distance method for evaluation of distance matrix, 'cosine' is advised ('euclidean' is the only option for 'ward' clustering)
            clustering_method (str = 'complete'): hierarchial clustering linkage method, advised: 'complete', 'average' and 'ward', for other options check: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage            
            save_dir (str = './experiment_outputs/loce_clustering'): directory to save results
            cluster_linkage_threshold (float = 3.0): cluster selection by linkage distance (similar to color_threshold of scipy.cluster.hierarchy.dendrogram) 
        """
        self.selected_tag_ids_and_colors = selected_tag_ids_and_colors
        self.selected_tag_id_names = selected_tag_id_names
        self.selected_tag_ids = list(selected_tag_ids_and_colors.keys())
        self.loce_layers = loce_layers
        self.max_samples_per_category = max_samples_per_category

        self.cluster_size_threshold = cluster_size_threshold
        self.cluster_purity_threshold = cluster_purity_threshold
        self.distance_metric = distance_metric
        self.clustering_method = clustering_method

        self.save_dir = save_dir

        self.cluster_linkage_threshold = cluster_linkage_threshold

        self.cluster_infos: List[LoCEMultilayerClusterInfo] = None  # meta info for each cluster (centroid, category counts, etc.) obtained from the last run of cluster_loces()

        self.cluster_labeled_loce: Tuple[List[LoCEMultilayerStorage], List[int]] = None  # loces storages with cluster labels for kNN, tuple - (LoCEs, cluster int labels)

    def cluster_loces(self,
                      loce_storages_dict: Dict[int, List[LoCEMultilayerStorage]],
                      save_imgs: bool = True,
                      save_proj: bool = True,
                      model_categories: Dict[int, str] = None,
                      clustering_type: Literal['adaptive', 'linkage', 'size'] = 'adaptive',
                      verbose: str = True
                      ) -> List[List[LoCEMultilayerStorage]]:
        """
        Perform hierarchial clustering of LoCEs

        Args:
            loce_storages (Dict[int, List[LoCEMultilayerStorage]]): LoCE storages per category for clustering

        Kwargs:
            save_imgs (bool = True): save or not visual results (images) to self.save_dir/loce_clustered_imgs_tiles/
            save_projs (bool = True): save or not visual results (projections) to self.save_dir/loce_clustered_proj_tiles/
            model_categories (str = None): different models may have different id-label mappings, needed for correct visualization of bboxes
            clustering_type (Literal['adaptive', 'linkage', 'size'] = 'adaptive'): 'adaptive' uses self.cluster_size_threshold and self.cluster_purity_threshold, 'linkage' uses self.cluster_linkage_threshold, 'size' uses self.cluster_size_threshold

        Returns:
            clustered_loce_storages (List[List[LoCEMultilayerStorage]]): list of lists of LoCEMultilayerStorage, where external list is clusters, internal lists are leaf LoCEMultilayerStorage
        """
        #log_info("Clustering...")
        # clustering
        selected_loce_storages, linkage, cluster_leaf_list, cluster_node_list, cluster_node_ids = self._cluster(loce_storages_dict, clustering_type)

        # reorder storages according to cluster_leaf_list
        clustered_loce_storages = self._reorder_selected_loce_storages_to_clusters(selected_loce_storages, cluster_leaf_list)

        # get clusters meta information (centroids, categories count, etc)
        self.cluster_infos = self._loce_cluster_infos(clustered_loce_storages)

        # estimate clustering quality metric
        n_categories = len(loce_storages_dict)
        clustering_state = self._estimate_clustering_state(verbose)

        # loces and their cluster labels (for kNN)
        self.cluster_labeled_loce = (selected_loce_storages, self._loce_cluster_labels(cluster_leaf_list))
        
        if save_imgs:
            imgs_dir = os.path.join(self.save_dir, 'loce_clustered_imgs_tiles')
            rmdir(imgs_dir)
            save_cluster_imgs_as_tiles(clustered_loce_storages, self.selected_tag_id_names, self.selected_tag_ids_and_colors, imgs_dir, (256, 192), model_categories)

        if save_proj:
            proj_dir = os.path.join(self.save_dir, 'loce_clustered_proj_tiles')
            rmdir(proj_dir)
            save_cluster_proj_as_tiles(clustered_loce_storages, self.selected_tag_id_names, self.selected_tag_ids_and_colors, proj_dir, (256, 192), model_categories, self.loce_layers)

        return clustered_loce_storages
    
    def get_clustering_state(self,
                             loce_storages_dict: Dict[int, List[LoCEMultilayerStorage]],
                             clustering_type: Literal['adaptive', 'linkage', 'size'] = 'adaptive'
                             ) -> Tuple[float, int]:

        selected_loce_storages, linkage, cluster_leaf_list, cluster_node_list, cluster_node_ids = self._cluster(loce_storages_dict, clustering_type)

        clustered_loce_storages = self._reorder_selected_loce_storages_to_clusters(selected_loce_storages, cluster_leaf_list)

        self.cluster_infos = self._loce_cluster_infos(clustered_loce_storages)

        purity, n_clusters = self._estimate_clustering_state(False)
        return purity, n_clusters
    
    @staticmethod
    def _reorder_selected_loce_storages_to_clusters(selected_loce_storages: Iterable[LoCEMultilayerStorage],
                                                    cluster_leaf_list: Iterable[Iterable[int]]
                                                    ) -> List[List[LoCEMultilayerStorage]]:
        """
        Wraps list of LoCE storages to double list, where external list is clusters, internal - cluster leafs (samples)

        Args:
            selected_loce_storages (Iterable[LoCEMultilayerStorage]): LoCE storages
            cluster_leaf_list (Iterable[Iterable[int]]): cluster leafs indices

        Retrns:
            clustered_loce_storages (List[List[LoCEMultilayerStorage]]): LoCE storages in cluster-like wrapping, external list is clusters, internal - cluster leafs (samples)
        """
        clustered_loce_storages = []
        for cluster in cluster_leaf_list:
            loce_storages_in_cluster = []
            for leaf in cluster:
                loce_storages_in_cluster.append(selected_loce_storages[leaf])
            clustered_loce_storages.append(loce_storages_in_cluster)

        return clustered_loce_storages

    def _loce_cluster_infos(self,
                            clustered_loce_storages: List[List[LoCEMultilayerStorage]],
                            ) -> List[LoCEMultilayerClusterInfo]:
        """
        Get list of centroids obtained from all clustered LoCEs

        Args:
            clustered_loce_storages (List[List[LoCEMultilayerStorage]]): LoCE storages per category for clustering

        Returns:
            cluster_infos (List[LoCEMultilayerClusterInfo]): list of meta information for each cluster
        """ 
        cluster_infos = [LoCEMultilayerClusterInfo(storages) for storages in clustered_loce_storages]

        for i, ci in enumerate(cluster_infos):
            prob_str = ', '.join([f'P({self.selected_tag_id_names[cat]}) = {prob:.2f}' for cat, prob in ci.cluster_category_probabilities.items()])
            # log_info(f'Cluster {i}: {prob_str}')

        return cluster_infos
    
    def _estimate_clustering_state(self, verbose=True) -> Tuple[float, int]:
        """
        Estimate the state after clustering

        State is estimated:
        maximize purity -> homogenous clusters -> mean purity
        minimize number of clusters -> less confusion in the feature space -> 1 / number of clusters

        Returns:
            clustering_state (float): clustering state metric
        """
        top_cat_counts = [ci.get_cluster_top_category_count() for ci in self.cluster_infos]
        cluster_sizes = [len(ci) for ci in self.cluster_infos]
        purity = sum(top_cat_counts) / sum(cluster_sizes)
        n_clusters = len(self.cluster_infos)

        layers_string = ' & '.join(self.loce_layers)
        
        if verbose:
            log_info(f'Clustering layers: {layers_string}, Purity: {purity:.3f}, Number of clusters: {n_clusters}')

        return purity, n_clusters

    @staticmethod
    def _loce_cluster_labels(cluster_leaf_list: List[List[int]]) -> List[int]:
        """
        Get LoCE cluster labels in original order (obtained from clustered leafes)

        Args:
            cluster_leaf_list (List[List[int]]): cluster leafs indices

        Returns:
            loce_cluster_labels (List[int]): LoCE cluster labels in original order
        """
        n_leafs = sum([len(i) for i in cluster_leaf_list])

        loce_cluster_labels = []

        for i in range(n_leafs):
            for cluster_idx, cluster_leafs in enumerate(cluster_leaf_list):
                if i in cluster_leafs:
                    loce_cluster_labels.append(cluster_idx)
        
        return loce_cluster_labels
    
    def cluster_stats(self) -> None:

        loces = [{l: cluster.get_centroid_loce(l) for l in self.loce_layers} for cluster in self.cluster_infos]
        fig, axs = plt.subplots(1, len(self.loce_layers), figsize=(20, 5))
        for j, l in enumerate(self.loce_layers):
            for i, loce in enumerate(loces):            
                axs[j].plot(loce[l], label=f'cluster {i}')
            axs[j].legend()

        plt.savefig(os.path.join(self.save_dir, 'filter_freqs.pdf'), bbox_inches='tight', pad_inches=0)
        plt.close()

        mlloces = [cluster.get_centroid_mlloce(self.loce_layers) for cluster in self.cluster_infos]

        hm1 = pairwise_distances(mlloces, metric='euclidean')
        hm2 = pairwise_distances(mlloces, metric='cosine')

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        sns.heatmap(hm1, ax=axs[0], cmap='vlag')
        sns.heatmap(hm2, ax=axs[1], cmap='vlag')
        #sns.clustermap(hm2, cmap='vlag')

        plt.savefig(os.path.join(self.save_dir, 'cluster_stats.pdf'), bbox_inches='tight', pad_inches=0)
        plt.close()

    def cluster_filter_frequencies_analysis(self, n=5) -> None:

        cluster_loces = [{l: cluster.get_centroid_loce(l) for l in self.loce_layers} for cluster in self.cluster_infos]
        
        for cluster_id, loce_dict in enumerate(cluster_loces):
            print(cluster_id)
            for l in self.loce_layers:
                loce = loce_dict[l]
                argsorted_loce = np.argsort(loce)[::-1]
                argsorted_loce_abs = np.argsort(np.abs(loce))[::-1]

                top_filters = argsorted_loce[:n]
                top_filters_abs = argsorted_loce_abs[:n]
                print('orig', l, top_filters.tolist())
                print('abs', l, top_filters_abs.tolist())
    
    def predict_loces_with_centroids(self,
                                     loce_storages_dict: Dict[int, List[LoCEMultilayerStorage]],
                                     save_imgs: bool = True
                                     ) -> Dict[int, List[Dict[int, float]]]:
        """
        Predict new LoCEs with cluster centroids

        Args:
            loce_storages (Dict[int, List[LoCEMultilayerStorage]]): LoCE storages per category for assignment to new clusters

        Kwargs:
            save_imgs (bool): save or not visual results to self.save_dir/loce_predictions_centroids/

        Returns:
            predictions (Dict[int, List[Dict[int, float]]]): predictions for every LoCE storage per category
        """
        log_info("Predicting with centroids...")

        if self.cluster_infos is None:
            raise ValueError("Train clusterer first, run cluster_loces() first")
        else:
            predictions = dict()
            predictions_cluster_idxs = dict() 
            for c in sorted(list(loce_storages_dict.keys())):
                category_storages = loce_storages_dict[c]

                category_loces = self._get_mlloce(category_storages)

                cluster_centroids = np.array([i.get_centroid_mlloce(self.loce_layers) for i in self.cluster_infos])

                dist_matr = pairwise_distances(cluster_centroids, category_loces, metric=self.distance_metric)

                predictions_cluster_idxs[c] = np.argmin(dist_matr, axis=0)
                predictions[c] = [self.cluster_infos[i].cluster_category_probabilities for i in predictions_cluster_idxs[c]]
                
            if save_imgs:
                save_dir = os.path.join(self.save_dir, 'loce_predictions_centroids')
                self._save_images_for_predictions(loce_storages_dict, predictions_cluster_idxs, save_dir, self._generate_prediction_string_centroid)

            return predictions
        
    def predict_loces_with_knn(self,
                               loce_storages_dict: Dict[int, List[LoCEMultilayerStorage]],
                               k: int = 5,
                               save_imgs: bool = True
                               ) -> Dict[int, List[Dict[int, float]]]:
        """
        Predict new LoCEs with k-nearest neigbours

        Args:
            loce_storages (Dict[int, List[LoCEMultilayerStorage]]): LoCE storages per category for assignment to new clusters

        Kwargs:
            k (int): k neighbours to use
            save_imgs (bool): save or not visual results to self.save_dir/loce_predictions_knn/

        Returns:
            predictions (Dict[int, List[Dict[int, float]]]): predictions for every LoCE storage per category (probabilities of true labels of nearest neigbours)
        """
        log_info("Predicting with kNN...")

        if self.cluster_labeled_loce is None:
            raise ValueError("Train clusterer first, run cluster_loces() first")
        else:
            # knn data
            neigbour_loces = self._get_mlloce(self.cluster_labeled_loce[0])
            neigbour_loces_true_labels = np.array([loce.segmentation_category_id for loce in self.cluster_labeled_loce[0]])

            knn = KNeighborsClassifier(k, metric=self.distance_metric)
            knn.fit(neigbour_loces, neigbour_loces_true_labels)

            predictions = dict()
            for c in sorted(list(loce_storages_dict.keys())):
                category_storages = loce_storages_dict[c]

                category_loces = self._get_mlloce(category_storages)

                probs = knn.predict_proba(category_loces)

                category_pred_dict = [{int(c): float(p) for c, p in zip(knn.classes_, prob)} for prob in probs]

                predictions[c] = category_pred_dict

            if save_imgs:
                save_dir = os.path.join(self.save_dir, 'loce_predictions_knn')
                self._save_images_for_predictions(loce_storages_dict, predictions, save_dir, self._generate_prediction_string_knn)

            return predictions

    def _save_images_for_predictions(self,
                                     loce_storages_dict: Dict[int, List[LoCEMultilayerStorage]],
                                     predictions: Union[Dict[int, List[int]], Dict[int, List[Dict[int, float]]]],
                                     img_out_path: str,
                                     title_string_fn: Callable[[Union[int, Dict[int, float]]], str]
                                     ) -> None:
        """
        Save images of centroid predictions of LoCEs

        Args:
            loce_storages (Dict[int, List[LoCEMultilayerStorage]]): LoCE storages per category for assignment to new clusters
            predictions (Union[Dict[int, List[int]], Dict[int, List[Dict[int, float]]]]): cluster or kNN predictions for every LoCE storage per category
            img_out_path (str): output directory for images
            title_string_fn (Callable[[Union[int, Dict[int, float]]], str]): function that takes single predicion as argument and produces an image title string from it
        """
        mkdir(img_out_path)

        for c in sorted(list(loce_storages_dict.keys())):
            category_storages = loce_storages_dict[c]
            category_predictions = predictions[c]

            for storage, pred in zip(category_storages, category_predictions):

                img_load_path = storage.image_path
                img_name = os.path.basename(img_load_path)
                img_name_base, ext = os.path.splitext(img_name)
                save_img_name = f"{img_name_base}_{c}{ext}"
                img_save_path = os.path.join(img_out_path, save_img_name)
                img_seg = storage.segmentation

                projections = [storage.loce_storage[l].projection for l in self.loce_layers]
                avg_projection = combine_masks(projections)

                img_str = title_string_fn(pred)

                # load image, create plot, add title and save
                img = Image.open(img_load_path).convert('RGB')
                seg_arr = np.array(blend_imgs(img, get_colored_mask(img_seg, [1], mask_value_multiplier=255)))
                proj_arr = apply_heatmap(np.array(img), avg_projection / 255.)

                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(seg_arr)
                axs[0].axis('off')
                axs[0].set_title('segmentation')

                axs[1].imshow(proj_arr)
                axs[1].axis('off')
                axs[1].set_title('avg. projection')

                plt.suptitle(img_str)
                #plt.tight_layout()
                plt.savefig(img_save_path, bbox_inches='tight', pad_inches=0)
                plt.close()

    def _generate_prediction_string_centroid(self,
                                             pred: int,
                                             ) -> str:
        """
        Generate a prediction string which contains information about probabilities of true class label assignments in clusters

        Args:
            pred (int): predicted cluster label

        Returns:
            pred_str (str): prediction string
        """
        cluster_category_probs = self.cluster_infos[pred].cluster_category_probabilities
        pred_str = f'Cluster {pred}:'
        for cat, prob in cluster_category_probs.items():
            pred_str += f' {self.selected_tag_id_names[cat]} - {prob:.2f}'
        
        return pred_str

    def _generate_prediction_string_knn(self,
                                        pred: Dict[int, float]
                                        ) -> str:
        """
        Generate a prediction string which contains information about probabilities of true class label assignments in clusters

        Args:
            pred (Dict[int, float]): probabilities of labels of nearest neighbors

        Returns:
            pred_str (str): prediction string
        """
        labels = sorted(list(pred.keys()))

        labels_total = sum([pred[l] for l in labels])

        pred_str = f'kNN category probs.:'
        for l in labels:
            if pred[l] != 0.:
                pred_str += f' {self.selected_tag_id_names[l]} - {(pred[l]/ labels_total):.2f}'
        
        return pred_str
    
    def _cluster(self,
                 loce_storages: Dict[int, List[LoCEMultilayerStorage]],
                 clustering_type: Literal['adaptive', 'linkage', 'size']
                 ) -> Tuple[List[LoCEMultilayerStorage], np.ndarray, List[List[int]], List[List[int]], List[int]]:
        """
        Perform hierarchial clustering of LoCEs

        Args:
            loce_storages (Dict[int, List[LoCEMultilayerStorage]]): LoCE storages per category for clustering,
            clustering_type (Literal['adaptive', 'linkage', 'size']): 'adaptive' uses self.cluster_size_threshold and self.cluster_purity_threshold, 'linkage' uses self.cluster_linkage_threshold, 'size' uses self.cluster_size_threshold
        
        Returns:
            selected_loce_storages (List[LoCEMultilayerStorage]) - LoCE storages used for clustering
            linkage (np.ndarray) - scipy linkage array
            cluster_leaf_list (List[List[int]]): list of lists of leaf ids, where external list is clusters, internal lists are leaf ids
            cluster_node_list (List[List[int]]): list of lists of tree node ids, where external list is clusters, internal lists are tree node ids
            cluster_node_ids (List[int]): list of ids of cluster branch nodes
        """
        # select LoCEs for clustering
        #loce_data = []  # storage for only 
        selected_loce_storages = []  # selected samples for clustering
        #labels_int = []  # true category labels
        for c in self.selected_tag_ids:
            category_storages = loce_storages[c]
            for loce_storage in category_storages[:self.max_samples_per_category]:
                selected_loce_storages.append(loce_storage)

        labels_int = [s.segmentation_category_id for s in selected_loce_storages]
        loce_data = self._get_mlloce(selected_loce_storages)

        # metric + method combinations:
        # average + cosine
        # complete + cosine
        # ward + euclidean
        # 'average' = mean of distances between all points in both connected clusters (distance between centroids)
        # 'complete' = max distance between two furthest points in both connected clusters
        # 'ward' (variance minimization) - dist = obtained state variance (sum of distances from samples to possible cluster centroids), of all possible cluster connections

        linkage = hierarchy.linkage(loce_data, method=self.clustering_method, metric=self.distance_metric)

        if clustering_type == 'adaptive':
            # run adaptive clustering with min_purity and min_cluster_size constraints
            cluster_leaf_list, cluster_node_list, cluster_node_ids = self._run_adaptive_clustering(hierarchy.to_tree(linkage), np.array(labels_int))
        elif clustering_type == 'size':
            # run adaptive clustering with and max_cluster_size constraint
            cluster_leaf_list, cluster_node_list, cluster_node_ids = self._run_size_clustering(hierarchy.to_tree(linkage), np.array(labels_int))
        elif clustering_type == 'linkage':
            cluster_leaf_list, cluster_node_list, cluster_node_ids = self._run_linkage_clustering(hierarchy.to_tree(linkage), np.array(labels_int))
        else:
            raise ValueError("wrong 'clustering_type', shall be one of: ['adaptive', 'linkage', 'size']")
        
        return selected_loce_storages, linkage, cluster_leaf_list, cluster_node_list, cluster_node_ids
    
    def _get_mlloce(self, storages: List[LoCEMultilayerStorage]) -> np.ndarray:
        """
        Get ML-LoCEs for given storages, self.loce_layers are used

        Args:
            storages (List[LoCEMultilayerStorage]): storages for ML-LoCEs extraction

        Returns:
            mlloces (np.ndarray): 2D-numpy-array of stacked and combined ML-LoCEs
        """
        mlloces = np.array([s.get_multilayer_loce(self.loce_layers) for s in storages])
        return mlloces
    
    @staticmethod
    def _get_subtree_labels_frequencies(subtree_leafs_ids: Iterable[int],
                                        true_labels: np.ndarray
                                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate frequencies of labels in the subtree (potential cluster)

        Args:
            subtree_leafs_ids (Iterable[int]): ids of subtree leafs
            true_labels (np.ndarray): all true labels

        Returns:
            unique_lables (np.ndarray): unique labels in the subtree (potential cluster)
            label_freqs (np.ndarray): frequencies of unique labels in the subtree (potential cluster)
        """
        labels_in_subtree = true_labels[subtree_leafs_ids]

        unique_lables, unique_counts = np.unique(labels_in_subtree, return_counts=True)

        label_freqs = unique_counts / len(labels_in_subtree)

        return unique_lables, label_freqs

    @staticmethod
    def _get_subtree_node_ids(linkage_tree: hierarchy.ClusterNode) -> Tuple[List[int], List[int]]:
        """
        Return linkage_tree node ids 

        Args:
            linkage_tree (hierarchy.ClusterNode): tree

        Returns:
            subtree_leaf_ids (List[int]): list with only leaf node ids
            subtree_node_ids (List[int]): list with all node ids
        """

        subtree_leaf_ids = []
        subtree_node_ids = []
        
        def traverse_and_gather_ids(subtree: hierarchy.ClusterNode):

            if subtree.is_leaf():
                subtree_leaf_ids.append(subtree.get_id())
                subtree_node_ids.append(subtree.get_id())
            else:
                subtree_node_ids.append(subtree.get_id())

                if subtree.left:
                    traverse_and_gather_ids(subtree.left)

                if subtree.right:
                    traverse_and_gather_ids(subtree.right)

        traverse_and_gather_ids(linkage_tree)

        return subtree_leaf_ids, subtree_node_ids

    def _run_adaptive_clustering(self,
                                 linkage_tree: hierarchy.ClusterNode,
                                 true_labels: np.ndarray,
                                 ) -> Tuple[np.ndarray, List[List[int]], List[List[int]], List[int]]:
        """
        Perform adaptive clustering with min_purity and min_cluster_size constraints
        
        Args:
            linkage_tree (hierarchy.ClusterNode): tree
            true_labels (np.ndarray): true labels of samples for purity estimation

        Return:
            cluster_leaf_list (List[List[int]]): list of lists of leaf ids, where external list is clusters, internal lists are leaf ids
            cluster_node_list (List[List[int]]): list of lists of tree node ids, where external list is clusters, internal lists are tree node ids
            cluster_node_ids (List[int]): list of ids of cluster branch nodes
        """
        cluster_leaf_list = []  # only leaf-nodes
        cluster_node_list = []  # all nodes
        cluster_node_ids = []  # only node ids of cluster branches

        def traverse_and_cluster(linkage_tree: hierarchy.ClusterNode):

            # if subtree is a leaf -> attach the leaf index as a single cluster
            if linkage_tree.is_leaf():
                cluster_leaf_list.append([linkage_tree.get_id()])
                cluster_node_list.append([linkage_tree.get_id()])
                cluster_node_ids.append(linkage_tree.id)

            # otherwise check size and purity thresholds
            else:
                
                #subtree_leaf_ids = linkage_tree.pre_order(lambda leaf: leaf.id)
                subtree_leaf_ids, subtree_node_ids = self._get_subtree_node_ids(linkage_tree)
                # purity check
                _, label_freqs = self._get_subtree_labels_frequencies(subtree_leaf_ids, true_labels)
                purity = max(label_freqs)

                # size check
                if linkage_tree.count <= self.cluster_size_threshold:
                    cluster_leaf_list.append(subtree_leaf_ids)
                    cluster_node_list.append(subtree_node_ids)
                    cluster_node_ids.append(linkage_tree.id)

                # if purity is high -> write to dict and exit recursion
                elif purity >= self.cluster_purity_threshold:
                    cluster_leaf_list.append(subtree_leaf_ids)
                    cluster_node_list.append(subtree_node_ids)
                    cluster_node_ids.append(linkage_tree.id)

                # if check were not passed -> decompose into subclusters / branches
                else:
                    if linkage_tree.left:
                        traverse_and_cluster(linkage_tree.left)

                    if linkage_tree.right:
                        traverse_and_cluster(linkage_tree.right)

        traverse_and_cluster(linkage_tree)

        return cluster_leaf_list, cluster_node_list, cluster_node_ids
    
    def _run_size_clustering(self,
                             linkage_tree: hierarchy.ClusterNode,
                             true_labels: np.ndarray,
                             ) -> Tuple[np.ndarray, List[List[int]], List[List[int]], List[int]]:
        """
        Perform clustering with cluster selection by max cluster size defined with self.cluster_size_threshold
        
        Args:
            linkage_tree (hierarchy.ClusterNode): tree
            true_labels (np.ndarray): true labels of samples for purity estimation

        Return:
            cluster_leaf_list (List[List[int]]): list of lists of leaf ids, where external list is clusters, internal lists are leaf ids
            cluster_node_list (List[List[int]]): list of lists of tree node ids, where external list is clusters, internal lists are tree node ids
            cluster_node_ids (List[int]): list of ids of cluster branch nodes
        """
        cluster_leaf_list = []  # only leaf-nodes
        cluster_node_list = []  # all nodes
        cluster_node_ids = []  # only node ids of cluster branches

        def traverse_and_cluster(linkage_tree: hierarchy.ClusterNode):

            # if subtree is a leaf -> attach the leaf index as a single cluster
            if linkage_tree.is_leaf():
                cluster_leaf_list.append([linkage_tree.get_id()])
                cluster_node_list.append([linkage_tree.get_id()])
                cluster_node_ids.append(linkage_tree.id)

            # otherwise check size threshold
            else:
                
                #subtree_leaf_ids = linkage_tree.pre_order(lambda leaf: leaf.id)
                subtree_leaf_ids, subtree_node_ids = self._get_subtree_node_ids(linkage_tree)

                # size check
                if linkage_tree.count <= self.cluster_size_threshold:
                    cluster_leaf_list.append(subtree_leaf_ids)
                    cluster_node_list.append(subtree_node_ids)
                    cluster_node_ids.append(linkage_tree.id)

                # if check was not passed -> decompose into subclusters / branches
                else:
                    if linkage_tree.left:
                        traverse_and_cluster(linkage_tree.left)

                    if linkage_tree.right:
                        traverse_and_cluster(linkage_tree.right)

        traverse_and_cluster(linkage_tree)

        return cluster_leaf_list, cluster_node_list, cluster_node_ids
    
    def _run_linkage_clustering(self,
                                linkage_tree: hierarchy.ClusterNode,
                                true_labels: np.ndarray,
                                ) -> Tuple[np.ndarray, List[List[int]], List[List[int]], List[int]]:
        """
        Perform linkage clustering with cluster selection by linkage distance
        
        Args:
            linkage_tree (hierarchy.ClusterNode): tree
            true_labels (np.ndarray): true labels of samples for purity estimation

        Return:
            cluster_leaf_list (List[List[int]]): list of lists of leaf ids, where external list is clusters, internal lists are leaf ids
            cluster_node_list (List[List[int]]): list of lists of tree node ids, where external list is clusters, internal lists are tree node ids
            cluster_node_ids (List[int]): list of ids of cluster branch nodes
        """
        cluster_leaf_list = []  # only leaf-nodes
        cluster_node_list = []  # all nodes
        cluster_node_ids = []  # only node ids of cluster branches

        def traverse_and_cluster(linkage_tree: hierarchy.ClusterNode):

            # if subtree is a leaf -> attach the leaf index as a single cluster
            if linkage_tree.is_leaf():
                cluster_leaf_list.append([linkage_tree.get_id()])
                cluster_node_list.append([linkage_tree.get_id()])
                cluster_node_ids.append(linkage_tree.id)

            # otherwise check distance thresholds
            else:
                
                #subtree_leaf_ids = linkage_tree.pre_order(lambda leaf: leaf.id)
                subtree_leaf_ids, subtree_node_ids = self._get_subtree_node_ids(linkage_tree)

                # distance check
                if linkage_tree.dist <= self.cluster_linkage_threshold:
                    cluster_leaf_list.append(subtree_leaf_ids)
                    cluster_node_list.append(subtree_node_ids)
                    cluster_node_ids.append(linkage_tree.id)

                # if check was not passed -> decompose into subclusters / branches
                else:
                    if linkage_tree.left:
                        traverse_and_cluster(linkage_tree.left)

                    if linkage_tree.right:
                        traverse_and_cluster(linkage_tree.right)

        traverse_and_cluster(linkage_tree)

        return cluster_leaf_list, cluster_node_list, cluster_node_ids

    def loce_dendogram(self,
                       loce_storages_dict: Dict[int, List[LoCEMultilayerStorage]],
                       clustering_type: Literal['adaptive', 'linkage', 'size'] = 'adaptive',
                       save_name: str = None,
                       ax: plt.Axes = None,
                       dataset_tag: Literal["mscoco", "voc"] = "mscoco"
                       ) -> None:
        """
        Draw a dendogram of hierarchial clustering of LoCEs

        Args:
            loce_storages (Dict[int, List[LoCEMultilayerStorage]]): LoCE storages per category for clustering

        Kwargs:
            clustering_type (Literal['adaptive', 'linkage', 'size'] = 'adaptive'): 'adaptive' uses self.cluster_size_threshold and self.cluster_purity_threshold, 'linkage' uses self.cluster_linkage_threshold, 'size' uses self.cluster_size_threshold
            save_name (str): name for plot saving in self.save_dir, f'loce_dendogram_{n_clusters}_{purity:.3f}.pdf' if not specified
        """
        log_info("Plotting dendogram...")

        def color_map_fn(idx: int,
                         cluster_node_list: List[List[int]],
                         mpl_color_strings: List[str],
                         no_clust_color: str) -> str:

            for j, c in enumerate(cluster_node_list):
                if idx in c:
                    return mpl_color_strings[j]

            return no_clust_color

        selected_loce_storages, linkage, cluster_leaf_list, cluster_node_list, cluster_node_ids = self._cluster(loce_storages_dict, clustering_type)

        clustered_loce_storages = self._reorder_selected_loce_storages_to_clusters(selected_loce_storages, cluster_leaf_list)
        self.cluster_infos = self._loce_cluster_infos(clustered_loce_storages)
        purity, n_clusters = self._estimate_clustering_state(verbose=False)

        if save_name is None:
            save_name = f'loce_dendogram_{n_clusters}_{purity:.3f}.pdf'

        # color strings initialiaztion
        # mpl_color_strings = ['#3e3636', '#5d5555', '#716868', '#999090', '#b9b0b0'] * 50
        mpl_color_strings = ['#3e3636', '#716868', '#b9b0b0'] * 1000
        no_clust_color = '#efe5e5'
        leaf_colors = []
        [leaf_colors.extend([cs] * len(cll)) for cs, cll in zip(mpl_color_strings, cluster_leaf_list)]

        labels_int = [s.segmentation_category_id for s in selected_loce_storages]

        # plot
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(20, 2))  # (20, 3) for confusion plot
        dendogram = hierarchy.dendrogram(linkage, link_color_func=lambda k: color_map_fn(k, cluster_node_list, mpl_color_strings, no_clust_color), ax=ax)

        order = dendogram['leaves']
        reordered_labels_char = np.array([self.selected_tag_ids_and_colors[i] for i in labels_int])[order]

        max_y = linkage[:, 2].max()
        underline_1_y = max_y * 0.02
        underline_2_y = max_y * 0.05
        #y_bot_lim = min(max_y * 0.07, np.ceil(underline_2_y))
        y_bot_lim = min(max_y * 0.05, np.ceil(underline_1_y))

        # cluster_labels
        # true labels
        x_min = np.min(np.array(dendogram['icoord']))
        x_max = np.max(np.array(dendogram['icoord']))
        scaled_positions = np.linspace(x_min, x_max, len(labels_int))
        ax.scatter(scaled_positions, [-underline_1_y for _ in labels_int], c=reordered_labels_char, marker='s')

        ax.set_ylim(bottom=-y_bot_lim)

        #ax.plot(0, self.cluster_linkage_threshold, len(labels_int) * 10, self.cluster_linkage_threshold, linestyle = '--', c='black')
        # remove x labels with sample ids
        ax.set_xticks([])
        ax.set_xticklabels([])

        ax.text(0.995, 0.98, f'Purity = {purity:.3f}', transform=ax.transAxes, ha='right', va='top', fontsize=20)

        # add cluster markers
        cluster_node_ids_changed = {(i - len(linkage) - 1): j for j, i in enumerate(cluster_node_ids)}
        # non-leafes - non-negative idxs
        ii = np.argsort(np.array(dendogram['dcoord'])[:, 1])
        for j, (icoord, dcoord) in enumerate(zip(dendogram['icoord'], dendogram['dcoord'])):
            x = 0.5 * sum(icoord[1:3])
            y = dcoord[1]
            ind = np.nonzero(ii == j)[0][0]
            if ind in cluster_node_ids_changed.keys():
                cluster_id = cluster_node_ids_changed[ind]
                ax.plot(x, y, marker='X', color='black', markersize=10)
                ax.annotate(f"{cluster_id}", (x-15, y), va='bottom', ha='right', fontsize=15)
        # leafes - negative idxs
        for idx, cluster_id in cluster_node_ids_changed.items():
            if idx < 0:
                # leaf_idx = cluster_leaf_list[cluster_id][0]
                leaf_idx = idx + len(linkage) + 1
                leaf_pos = (10 * sum([len(cluster_leaf_list[i]) for i in range(cluster_id)])) + 5
                ax.plot(leaf_pos, 0, marker='X', color='black', markersize=10)
                ax.annotate(f"{cluster_id}", (leaf_pos-15, 0), va='bottom', ha='right', fontsize=15)
        
        
        if dataset_tag == "voc":
            markers = VOC_MARKERS
        else:
            markers = MSCOCO_MARKERS

        selected_markers = {i: markers[i] if i in markers else 's' for i in self.selected_tag_ids}
        legend_dict = {self.selected_tag_ids_and_colors[i]: (self.selected_tag_id_names[i], selected_markers[i])  for i in self.selected_tag_ids}

        # Create a custom legend with square markers and category labels
        legend_handles = [plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor=color, markersize=15, label=label) for color, (label, marker) in legend_dict.items()]
        
        if len(legend_dict) >= 10:
            ncols = len(legend_dict) // 2 + len(legend_dict) % 2
        else:
            ncols = len(legend_dict)

        # Add the custom legend under the plot
        ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=ncols, frameon=False, handletextpad=0.0, columnspacing=0.4, fontsize=15)

        if ax is None:
            mkdir(self.save_dir)
            plt.savefig(os.path.join(self.save_dir, save_name))
            plt.close()

    def loce_clustermap(self,
                        loce_storages_dict: Dict[int, List[LoCEMultilayerStorage]],
                        save_name: str = 'loce_clustermap.pdf',
                        clustering_type: Literal['adaptive', 'linkage', 'size'] = 'adaptive'
                        ) -> None:
        """
        Draw a dendogram of hierarchial clustering of LoCEs

        Args:
            loce_storages (Dict[int, List[LoCEMultilayerStorage]]): LoCE storages per category for clustering

        Kwargs:
            save_name (str): name for plot saving in self.save_dir
            clustering_type (Literal['adaptive', 'linkage', 'size'] = 'adaptive'): 'adaptive' uses self.cluster_size_threshold and self.cluster_purity_threshold, 'linkage' uses self.cluster_linkage_threshold, 'size' uses self.cluster_size_threshold
        """
        log_info("Plotting clustermap...")

        def color_map_fn(idx: int,
                         cluster_node_list: List[List[int]],
                         mpl_color_strings: List[str],
                         no_clust_color: str) -> str:

            for j, c in enumerate(cluster_node_list):
                if idx in c:
                    return mpl_color_strings[j]

            return no_clust_color

        selected_loce_storages, linkage, cluster_leaf_list, cluster_node_list, cluster_node_ids = self._cluster(loce_storages_dict, clustering_type)

        # color strings initialiaztion
        mpl_color_strings = ['dimgrey', 'darkgrey'] * 1000
        no_clust_color = 'lightgrey'
        labels_int = [s.segmentation_category_id for s in selected_loce_storages]
        labels_char = [self.selected_tag_ids_and_colors[i] for i in labels_int]

        loce_data = self._get_mlloce(selected_loce_storages)

        dendogram = hierarchy.dendrogram(linkage, link_color_func=lambda k: color_map_fn(k, cluster_node_list, mpl_color_strings, no_clust_color), no_plot=True)

        distance_map = pairwise_distances(loce_data, metric=self.distance_metric)

        sns.clustermap(distance_map, cmap='vlag', row_linkage=linkage, col_linkage=linkage, row_colors=labels_char, col_colors=labels_char, tree_kws={'colors': dendogram['color_list']})
        plt.savefig(os.path.join(self.save_dir, save_name))
        plt.close()

    def loce_cluster_projection(self,
                                activations_extractor: LoCEActivationsTensorExtractor,
                                img_folder: str,
                                img_name: str,
                                selected_clusters_id: Iterable[int] = None,
                                selected_bts: Iterable[float] = [0.5],
                                normalized_loces: bool = False,
                                dataset_tag: Literal["mscoco", "voc"] = "mscoco",
                                image_loader: ImageLoader = ImageLoader()
                                ) -> None:
        """
        Get projections of cluster LoCEs for input image. Saves image as './self.save_dir/projections_{img_name}'

        Args:
            activations_extractor (LoCEActivationsTensorExtractor): activations extractor
            img_folder (str): folder with image
            img_name (str): name of image
            

        Kwargs:
            selected_clusters_id: (Iterable[int] = None): selected clusters to build projections for 
            selected_bts (Iterable[float] = [0.75]): binarization thresholds to plot masks
        """
        log_info("Evaluating cluster projections...")

        if self.cluster_infos is None:
            raise ValueError("Train clusterer first, run cluster_loces() first")
        else:
            img_pil = image_loader.load_pil_img(img_folder, img_name)
            acts_dict, _ = activations_extractor.get_bchw_acts_preds_dict(img_pil)

            all_projections = []

            for i, cluster_info in enumerate(self.cluster_infos):
                if selected_clusters_id is not None:
                    if i not in selected_clusters_id:
                        continue
                cluster_projections = {l: get_projection(cluster_info.get_centroid_loce(l, normalized_loces), acts_dict[l][0].numpy()) for l in self.loce_layers}
                all_projections.append(cluster_projections)
            
            self._save_image_cluster_projections(img_pil, img_name, all_projections, dataset_tag, selected_clusters_id, selected_bts)

    def _save_image_cluster_projections2(self,
                                         img: Image.Image,
                                         img_name: str,
                                         all_projections: List[Dict[str, np.ndarray]],
                                         dataset_tag: Literal["mscoco", "voc"],
                                         selected_clusters_id: Iterable[int] = None,
                                         selected_bts: Iterable[float] = [0.5],
                                         ) -> None:
        """
        Save projections of cluster LoCEs for input image as './self.save_dir/projections/projections_{img_name}' and in separate images in './self.save_dir/projections/imgs/'

        Args:
            img_folder (str): folder with image
            img_name (str): name of image
            all_projections (List[Dict[str, np.ndarray]]): results of LoCEs projecting on input image

        Kwargs:
            selected_clusters: (selected_clusters = None): selected clusters to build projections for 
            selected_bts (Iterable[float] = [0.75]): binarization thresholds to plot masks
        """
        mkdir(os.path.join(self.save_dir, "projections", 'imgs'))

        img_size = img.size

        selected_layers = self.loce_layers
        if selected_clusters_id is not None:
            selected_clusters = [self.cluster_infos[sc] for sc in selected_clusters_id]
        else:
            selected_clusters = self.cluster_infos

        cluster_top_categories = []
        for sc in selected_clusters:
            cluster_top_category = max(sc.cluster_category_probabilities, key=sc.cluster_category_probabilities.get)
            cluster_top_categories.append(cluster_top_category)
        
        ncols = len(selected_clusters) + 1
        nrows = len(selected_bts) + 3

        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2), dpi=100)

        # original image: [0,0] axis
        axs[0,0].imshow(np.array(img))
        axs[0,0].axis('off')

        # index: continious masked images
        axs[1,0].text(0.5, 0.5, 'Heatmaps', ha='center', va='center')
        axs[1,0].set_frame_on(False)
        axs[1,0].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        # index: continious masks
        axs[2,0].text(0.5, 0.5, 'Continious masks', ha='center', va='center')
        axs[2,0].set_frame_on(False)
        axs[2,0].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        # index: BTs
        for y, bt in enumerate(selected_bts):
            axs[y+3,0].text(0.5, 0.5, f'BT={bt}', ha='center', va='center')
            axs[y+3,0].set_frame_on(False)
            axs[y+3,0].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        # header: cluster labels
        for x, c in enumerate(selected_clusters):
            cluster_category_probs = c.cluster_category_probabilities
            cluster_number = selected_clusters_id[x] if selected_clusters_id is not None else x
            pred_str = f'Cluster {cluster_number}:'
            for cat, prob in cluster_category_probs.items():
                pred_str += f'\n{self.selected_tag_id_names[cat]} - {prob:.2f}'
            axs[0,x+1].text(0.0, 0.0, pred_str, ha='left', va='bottom')
            axs[0,x+1].set_frame_on(False)
            axs[0,x+1].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        # combined projections
        for x, c in enumerate(selected_clusters):
            projections = all_projections[x]
            avg_projection = combine_masks(list(projections.values()))
            hm = apply_heatmap(np.array(img), avg_projection / 255.)
            #proj_img_arr = np.array(blend_imgs(img, get_colored_mask(avg_projection, [1], mask_value_multiplier=1)))
            axs[1,x+1].imshow(hm)
            axs[1,x+1].axis('off')
            hm_img = Image.fromarray(hm)
            tags_str = '_'.join(list(self.selected_tag_id_names.values()))
            if len(selected_clusters) == 1:
                x = 'single'
            hm_img.save(os.path.join(self.save_dir, "projections", 'imgs',  f"loce_heatmap_{tags_str}_{x}_{img_name}"))
            
        # combined projections
        for x, c in enumerate(selected_clusters):
            projections = all_projections[x]
            avg_projection = combine_masks(list(projections.values()))
            rgb_proj = get_rgb_binary_mask(avg_projection, img_size)
            axs[2,x+1].imshow(rgb_proj)
            axs[2,x+1].axis('off')
            hm_img = Image.fromarray(rgb_proj)
            tags_str = '_'.join(list(self.selected_tag_id_names.values()))
            if len(selected_clusters) == 1:
                x = 'single'
            hm_img.save(os.path.join(self.save_dir, "projections", 'imgs', f"loce_rgb_map_{tags_str}_{x}_{img_name}"))

        if dataset_tag == "voc":
            dataset_colors = VOC_CATEGORY_COLORS
        else:
            dataset_colors = MSCOCO_CATEGORY_COLORS

        for y, bt in enumerate(selected_bts):
            for x, (c, top_cat) in enumerate(zip(selected_clusters, cluster_top_categories)):
                temp_color = dataset_colors[top_cat]
                temp_color_rgb = tuple([int(c * 255) for c in colors.to_rgb(temp_color)])
                projections = all_projections[x]
                avg_projection = combine_masks(list(projections.values()))
                mask = avg_projection  / 255. 
                mask = normalize_0_to_1(mask)
                mask = resize(mask, (np.array(img).shape[0], np.array(img).shape[1]))
                binary_mask = mask >= bt
                masked_img = apply_mask(np.array(img), mask, bt, crop_around_mask=False)
                blended_img = blend_imgs(Image.fromarray(masked_img), img, 0.4)
                blended_img = add_countours_around_mask(np.array(blended_img), binary_mask.astype(np.uint8) * 255, countour_color=temp_color_rgb)
                #proj_img_arr = np.array(blend_imgs(img, get_colored_mask(projection, [1], mask_value_multiplier=1)))
                axs[y+3,x+1].imshow(blended_img)
                axs[y+3,x+1].axis('off')
                tags_str = '_'.join(list(self.selected_tag_id_names.values()))
                if len(selected_clusters) == 1:
                    x = 'single'
                Image.fromarray(blended_img).save(os.path.join(self.save_dir, "projections", 'imgs', f"loce_mask_{tags_str}_{x}_{bt}_{img_name}"))

        """# projections
        for y, l in enumerate(selected_layers):
            for x, c in enumerate(selected_clusters):
                projection = all_projections[x][l]
                hm = apply_heatmap(np.array(img), projection / 255.)
                #proj_img_arr = np.array(blend_imgs(img, get_colored_mask(projection, [1], mask_value_multiplier=1)))
                axs[y+2,x+1].imshow(hm)
                axs[y+2,x+1].axis('off')"""

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'projections_{os.path.basename(img_name)}.pdf'))
        plt.close()


    def _save_image_cluster_projections(self,
                                        img: Image.Image,
                                        img_name: str,
                                        all_projections: List[Dict[str, np.ndarray]],
                                        dataset_tag: Literal["mscoco", "voc"],
                                        selected_clusters_id: Iterable[int] = None,
                                        selected_bts: Iterable[float] = [0.5],
                                        ) -> None:
        """
        Save projections of cluster LoCEs for input image as './self.save_dir/projections/projections_{img_name}' and in separate images in './self.save_dir/projections/imgs/'

        Args:
            img_folder (str): folder with image
            img_name (str): name of image
            all_projections (List[Dict[str, np.ndarray]]): results of LoCEs projecting on input image

        Kwargs:
            selected_clusters: (selected_clusters = None): selected clusters to build projections for 
            selected_bts (Iterable[float] = [0.75]): binarization thresholds to plot masks
        """

        img_size = img.size

        selected_layers = self.loce_layers
        if selected_clusters_id is not None:
            selected_clusters = [self.cluster_infos[sc] for sc in selected_clusters_id]
        else:
            selected_clusters = self.cluster_infos

        cluster_top_categories = []
        for sc in selected_clusters:
            cluster_top_category = max(sc.cluster_category_probabilities, key=sc.cluster_category_probabilities.get)
            cluster_top_categories.append(cluster_top_category)
        
        ncols = len(selected_clusters)
        nrows = len(selected_bts)

        fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), dpi=100)

        if len(axs.shape) == 1: # single row
            axs = axs.reshape(1, -1)

        if dataset_tag == "voc":
            dataset_colors = VOC_CATEGORY_COLORS
        else:
            dataset_colors = MSCOCO_CATEGORY_COLORS

        for y, bt in enumerate(selected_bts):
            for x, (c, top_cat) in enumerate(zip(selected_clusters, cluster_top_categories)):
                temp_color = dataset_colors[top_cat]
                temp_color_rgb = tuple([int(c * 255) for c in colors.to_rgb(temp_color)])
                projections = all_projections[x]
                avg_projection = combine_masks(list(projections.values()))
                mask = avg_projection  / 255. 
                mask = normalize_0_to_1(mask)
                mask = resize(mask, (np.array(img).shape[0], np.array(img).shape[1]))
                binary_mask = mask >= bt
                masked_img = apply_mask(np.array(img), mask, bt, crop_around_mask=False)
                blended_img = blend_imgs(Image.fromarray(masked_img), img, 0.4)
                blended_img = add_countours_around_mask(np.array(blended_img), binary_mask.astype(np.uint8) * 255, countour_color=temp_color_rgb)
                #proj_img_arr = np.array(blend_imgs(img, get_colored_mask(projection, [1], mask_value_multiplier=1)))
                axs[y,x].imshow(blended_img)
                axs[y,x].axis('off')
                cluster_number = selected_clusters_id[x] if selected_clusters_id is not None else x
                axs[y,x].set_title(f"Cluster {cluster_number}")
                tags_str = '_'.join(list(self.selected_tag_id_names.values()))
                if len(selected_clusters) == 1:
                    x = 'single'
                #Image.fromarray(blended_img).save(os.path.join(self.save_dir, "projections", 'imgs', f"loce_mask_{tags_str}_{x}_{bt}_{img_name}"))

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'projections_{os.path.basename(img_name)}.jpg'))
        plt.close()


class LoCEClustererManyLoaders:

    def __init__(self,
                 clusterer: LoCEClusterer,
                 tagged_loaders: Dict[str, LoCEMultilayerStorageDirectoryLoader],
                 save_base_dir: str = './experiment_outputs/loce_clustering_many',
                 ) -> None:
        """
        Args:
            clusterer (LoCEClusterer): clusterer instance
            tagged_loaders (Dict[str, LoCEMultilayerStorageDirectoryLoader]): LoCE loaders with string tags
        
        Kwargs:
            save_base_dir (str = './experiment_outputs/loce_clustering_many'): base directory for saving, folders with tag names will be created for individual results
        """
        self.clusterer = clusterer
        self.tagged_loaders = tagged_loaders
        self.save_base_dir = save_base_dir

        self.common_files = self._get_common_files()

        # set loaders to have same files
        for storage in self.tagged_loaders.values():
            storage.pkl_file_names = self.common_files
        

    def _get_common_files(self) -> List[str]:
        """
        Get list of common files in all arrays for a fair comparison (due to different areas of segmentations and segmentation area thresholding, lists may be different)

        Returns:
            common_files (List[str]): common file names list across all self.tagged_loaders
        """
        files = [s.pkl_file_names for s in self.tagged_loaders.values()]

        common_files = set(files[0])

        for array in files[1:]:
            common_files.intersection_update(array)

        return sorted(list(common_files))
    
    def run_clustering(self, data_categories: List[int], train_split: float = 0.8):
        """
        Run clustering for all Loaders with given sampling parameters

        Args:
            allowed_categories (Iterable[int]): load only LoCE storage of allowed categories ids, None to load all

        Kwargs:            
            test_size (float = 0.2): size of 'test' split
        """
        for tag, loader in self.tagged_loaders.items():

            log_info(f"Loader with tag '{tag}'")

            self.clusterer.save_dir = os.path.join(self.save_base_dir, tag)

            loce_storages_train, loce_storages_test = loader.load_train_test_splits(data_categories, train_split)

            clustered_samples = self.clusterer.cluster_loces(loce_storages_train)

            #self.clusterer.cluster_stats()
            #self.clusterer.loce_cluster_projection(yolo5, in_dir_imgs, '000000338986.jpg')
            #predictions_knn = self.clusterer.predict_loces_with_knn(loce_storages_test, 11)
            #predictions_centroids = self.clusterer.predict_loces_with_centroids(loce_storages_test)

            self.clusterer.loce_clustermap(loce_storages_train)
            self.clusterer.loce_dendogram(loce_storages_train)

