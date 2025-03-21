'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

from xai_utils.logging import init_logger
init_logger()

from typing import Dict, List, Iterable, Tuple, IO
import os
import random
import math

import numpy as np
from scipy.stats import rankdata
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

from .loce_utils import get_projection, get_rgb_binary_mask, get_colored_mask, find_closest_square_rootable_number, draw_frame, MSCOCO_CATEGORIES, shorten_layer_name

from xai_utils.files import write_pickle, read_pickle, mkdir, blend_imgs, add_countours_around_mask


class LoCE():
    """
    Guided Concept Projection Vector (LoCE)

    Storage for single optimized instance
    """
    def __init__(self,
                 loce: np.ndarray,
                 loss: float,
                 projection: np.ndarray
                 ) -> None:
        """
        Args:
            loce (np.ndarray): projection vector with unit importance weights
            loss (float): final loss after LoCE optimization
            projection (np.ndarray): projection of LoCE with original activations used for optimization
        """
        
        self.loce = loce
        self.loss = loss
        self.projection = projection

    def project(self, activations: np.ndarray) -> np.ndarray:
        """
        Args:
            activations (np.ndarray): activations to project with stored LoCE vector
        """
        return get_projection(self.loce, activations)


class LoCEWithMetaInformation(LoCE):
    """
    Standalone instance of LoCE, which contains all meta information (layer, original image, segmentation, etc.)
    """
    def __init__(self, 
                 loce: np.ndarray, 
                 loss: float, 
                 projection: np.ndarray,
                 layer: str = None, 
                 image_path: str = None, 
                 image_predictions: np.ndarray = None, 
                 segmentation: np.ndarray = None,
                 segmentation_category_id: int = None
                 ) -> None:
        """
        Args:
            loce (np.ndarray): projection vector with unit importance weights
            loss (float): final loss after LoCE optimization
            projection (np.ndarray): projection of LoCE with original activations used for optimization
        
        Kwargs:
            layer (str = None): layer at which LoCE was optimized
            image_path (str = None): path to image used for optimization
            image_predictions (np.ndarray = None): predictions made for image used for optimization
            segmentation (np.ndarray = None): original segmentation used for optimization
            segmentation_category_id (int = None): segmentation class (object class, etc.)
        """
        super().__init__(loce, loss, projection)

        self.layer = layer
        self.image_path = image_path
        self.image_predictions = image_predictions
        self.segmentation = segmentation
        self.segmentation_category_id = segmentation_category_id


class LoCEMultilayerStorage():
    """
    Storage for LoCEs obtained in multiple layers.

    Contains LoCE instances and meta information (layer, original image, segmentation, etc.)
    """

    def __init__(self,
                 image_path: str,
                 image_predictions: np.ndarray,
                 segmentation: np.ndarray,
                 segmentation_category_id: int
                 ) -> None:
        """        
        Args:
            image_path (str = None): path to image used for optimization
            image_predictions (np.ndarray = None): predictions made for image used for optimization
            segmentation (np.ndarray = None): original segmentation used for optimization
            segmentation_category_id (int): segmentation class (object class, etc.)
        """
        self.image_path = image_path
        self.image_predictions = image_predictions
        self.segmentation = segmentation
        self.segmentation_category_id = segmentation_category_id

        # init storage, make it instance specific
        self.loce_storage: Dict[str, LoCE] = {}  # {layer: loce}

    def set_loce(self,
                 layer: str,
                 loce: LoCE
                 ) -> None:
        """
        Add LoCE instance to storage

        Args:
            layer (str): layer of LoCE
            loce (LoCE): instance of LoCE
        """
        self.loce_storage[layer] = loce

    def get_loce(self, layer: str) -> LoCE:
        """
        Return LoCE from given layer

        Args:
            layer (str): retrieval layer

        Returns:
            loce (LoCE): retrieved LoCE instance
        """
        return self.loce_storage[layer]

    def get_storage_layers(self) -> List[str]:
        """
        Return all layer names contained in storage

        Returns:
            storage_layers (List[str]): list of storage layer names
        """
        return list(self.loce_storage.keys())
    
    def get_multilayer_loce(self, layers_to_concatenate: List[str]) -> np.ndarray:
        """
        Return multi-layer LoCE (MLLoCE)

        Args:
            layers_to_concatenate (List[str]): list of layers for LoCE

        Returns:
            multilayer_loce (np.ndarray): stacked LoCEs for multiple layers
        """
        loces = [self.get_loce(l).loce for l in layers_to_concatenate]

        multilayer_loce = np.concatenate(loces, axis=0)

        return multilayer_loce


class LoCEMultilayerClusterInfo:

    def __init__(self,
                 storages: List[LoCEMultilayerStorage],
                 ) -> None:
        """
        Args:
            storages (List[LoCEMultilayerStorage]): storages in cluster
        """
        self.layers = storages[0].get_storage_layers()

        self.accumulated_loces = self._accumulate_loces(storages)

        self.cluster_img_paths = [s.image_path for s in storages]
        
        self.cluster_img_true_segm_categories = [s.segmentation_category_id for s in storages]

        self.cluster_category_counts = self._get_cluster_category_counts(storages)

        self.cluster_category_probabilities = self._get_cluster_category_probs()

    def _accumulate_loces(self,
                          storages: List[LoCEMultilayerStorage]
                          ) -> Dict[str, np.ndarray]:
        """
        Per-layer accumulate LoCEs in arrays

        Args:
            storages (List[LoCEMultilayerStorage]): LoCEs

        Returns:
            loces (Dict[str, np.ndarray]): dictionary of per-layer LoCEs
        """
        loces = dict()

        for l in self.layers:
            loces[l] = np.array([s.get_loce(l).loce for s in storages])
        
        return loces
    
    def get_centroid_loce(self,
                          layer: str,
                          normalized: bool = True
                          ) -> np.ndarray:
        """
        Retrieve a centroid for single layer

        Args:
            layer (str): layer name

        Kwargs:
            normalized (bool = True): normalize vectors before aggregation

        Returns:
            loce_centroid (np.ndarray): centroid for given layer
        """
        loces = self.accumulated_loces[layer]

        if normalized:
            loce_norms = np.linalg.norm(self.accumulated_loces[layer], axis=1, keepdims=True)
            loces = loces / loce_norms

        return loces.mean(axis=0)
    
    def get_cumulative_loce(self,
                            layer: str,
                            normalized: bool = True
                            ) -> np.ndarray:
        """
        Retrieve an accumulated LoCE for single layer

        Args:
            layer (str): layer name

        Kwargs:
            normalized (bool = True): normalize vectors before aggregation

        Returns:
            loce_centroid (np.ndarray): centroid for given layer
        """
        loces = self.accumulated_loces[layer]

        if normalized:
            loce_norms = np.linalg.norm(self.accumulated_loces[layer], axis=1, keepdims=True)
            loces = loces / loce_norms

        return loces.sum(axis=0)

    def get_centroid_mlloce(self,
                            layers: List[str],
                            normalized: bool = True
                            ) -> np.ndarray:
        """
        Retrieve a centroids for multiple layers

        Args:
            layers (List[str]): layers name

        Kwargs:
            normalized (bool = True): normalize vectors before aggregation

        Returns:
            mlloce_centroid (np.ndarray): concatenated centroids for given layers
        """
        mlloce_centroid = np.concatenate([self.get_centroid_loce(l, normalized) for l in layers])
        return mlloce_centroid

    @staticmethod
    def _get_cluster_category_counts(storages: List[LoCEMultilayerStorage]
                                     ) -> Dict[int, int]:
        """
        Get count of true segmentation categories for cluster

        Args:
            storages (List[LoCEMultilayerStorage]): LoCE storages

        Returns:
            categories_dict (Dict[int, int]): {category_id: category_counts}
        """
        categories_dict = dict()

        for storage in storages:
            storage_category = storage.segmentation_category_id

            if storage_category in categories_dict:
                categories_dict[storage_category] += 1
            else:
                categories_dict[storage_category] = 1

        return categories_dict
    
    def _get_cluster_category_probs(self) -> Dict[int, float]:
        """
        Get probabilities (count / n_samples) of true segmentation categories for cluster

        Args:
            storages (List[LoCEMultilayerStorage]): LoCE storages

        Returns:
            probabilities_dict (Dict[int, float]): {category_id: category_probability}
        """
        probabilities_dict = dict()

        categories = sorted(list(self.cluster_category_counts.keys()))
        samples_total = sum([self.cluster_category_counts[c] for c in categories])

        probabilities_dict = {c: self.cluster_category_counts[c]/ samples_total for c in categories}

        return probabilities_dict
    
    def get_cluster_top_category_prob(self) -> float:
        """
        Get top category prob of the cluster

        Returns:
            purity (float): cluster top category prob
        """
        prob = max(self.cluster_category_probabilities.values())

        return prob
    
    def get_cluster_top_category_count(self) -> int:
        """
        Get top category count of the cluster

        Returns:
            purity (float): cluster top category count
        """
        count = max(self.cluster_category_counts.values())

        return count
    
    def __len__(self):
        return len(self.cluster_img_paths)


class LoCEMultilayerStorageSaver:

    def __init__(self, working_directory: str) -> None:
        """
        Args:
            working_directory (str): working directory for input-output of LoCEMultilayerStorages
        """
        self.working_directory = working_directory

    def get_loce_storage_path_for_img_name(self,
                                           image_name: str,
                                           category_id: int) -> str:
        """
        Generate image for LoCE storage from image and category_id

        Args:
            image_name (str): image name
            category_id (int): category of object used for optimization
        """
        image_name_no_ext = image_name.split(".")[0]

        image_name_no_ext_with_prefix = '_'.join([str(category_id), image_name_no_ext])

        image_path_no_ext_with_prefix = os.path.join(self.working_directory, image_name_no_ext_with_prefix)

        out_path_pkl = image_path_no_ext_with_prefix + ".pkl"  # correct file name
        out_path_err = image_path_no_ext_with_prefix + ".err"  # error file name

        return out_path_pkl, out_path_err

    def save(self, loce_storage: LoCEMultilayerStorage, save_path: str = None) -> None:
        """
        Saves loce_storage to {save_path}

        Args:
            loce_storage (LoCEMultilayerStorage): storage to save

        Kwargs:
            save_path (str): saving path, if not given - evaluate path with self.get_loce_storage_path_for_img_name()
        """
        if save_path is None:
            save_path = self.get_loce_storage_path_for_img_name(os.path.basename(loce_storage.image_path), loce_storage.segmentation_category_id)

        write_pickle(loce_storage, save_path)


class LoCEMultilayerStorageDirectoryLoader:

    def __init__(self,
                 working_directory: str,
                 seed: int = None,
                 min_seg_area: float = 0.0,
                 max_seg_area: float = 1.0
                 ) -> None:
        """
        Args:
            working_directory (str): working directory for input-output of LoCEMultilayerStorages

        Kwargs:
            seed (int = None) seed for data sampling
            min_seg_area (float): minimal allowed segmentation area, LoCEs with smaller segmentation will be ignored
            max_seg_area (float): maximal allowed segmentation area, LoCEs with larger segmentation will be ignored
        """
        self.working_directory = working_directory
        self.seed = seed
        random.seed(seed)

        self.min_seg_area = min_seg_area
        self.max_seg_area = max_seg_area

        self.pkl_file_names = self._select_pkl_files()

        self._filter_files_by_segmentation_size()

    def _select_pkl_files(self) -> List[str]:
        """
        Find file names of pickles with LoCEs

        Returns:
            selected_files (List[str]): filtered files by extension (only .pkl)
        """
        all_files_in_dir = sorted(os.listdir(self.working_directory))

        selected_files = []

        for fn in all_files_in_dir:

            if fn.split('.')[-1] == 'pkl':
                selected_files.append(os.path.join(self.working_directory, fn))
            else:
                continue

        return selected_files
    
    def _filter_files_by_segmentation_size(self) -> None:
        """
        Additional filtering of LoCEs by min and max segmentation areas
        """
        filtered_files = []

        for fn in self.pkl_file_names:
            loce: LoCEMultilayerStorage = read_pickle(fn)

            seg_mask = loce.segmentation
            seg_area = seg_mask.sum() / seg_mask.size

            if (seg_area > self.min_seg_area) and (seg_area < self.max_seg_area):
                filtered_files.append(fn)
            
        self.pkl_file_names = filtered_files

    def _categorize_files(self,
                          ) -> Dict[int, List[str]]:

        """
        Find file names of pickles with LoCEs

        Returns:
            (Dict[int, List[str]]) dictionary with LoCE file names - {tag_id: [file_name]}
        """
        file_names = self.pkl_file_names
        
        if not self.seed is None:
            random.shuffle(file_names)

        selected_files = dict()

        for fn in file_names:
            base_fn = os.path.basename(fn)
            category, name = base_fn.split('_', 1)
            
            # init key for current category
            if int(category) not in list(selected_files.keys()):
                selected_files[int(category)] = []
            
            selected_files[int(category)].append(fn)

        return selected_files
    
    def load(self,
             allowed_categories: Iterable[int] = None
             ) -> Dict[int, List[LoCEMultilayerStorage]]:
        """
        Get dictionary of LoCEMultilayerStorages lists per category

        Kwargs:
            allowed_categories (Iterable[int]): load only LoCE storage of allowed categories ids, None to load all

        Returns:
            loce_storages_dict (Dict[int, List[LoCEMultilayerStorage]]) LoCE storages per category
        """

        loce_file_names = self._categorize_files()

        if allowed_categories is not None:
            loce_storages_dict = {category: [] for category in sorted(allowed_categories)}
        else:
            loce_storages_dict = {category: [] for category in sorted(loce_file_names.keys())}

        for category in loce_storages_dict.keys():

            if category not in loce_file_names:
                continue

            for file_name in loce_file_names[category]:
                loce_storages_dict[category].append(read_pickle(file_name))

        #if allowed_categories is not None:
        #    loce_storages_dict = {t: [read_pickle([os.path.join(self.working_directory, f)]) for f in loce_file_names[t]] for t in allowed_categories}
        #else:
        #    loce_storages_dict = {t: [read_pickle([os.path.join(self.working_directory, f)]) for f in loce_file_names[t]] for t in sorted(loce_file_names.keys())}

        return loce_storages_dict
    
    def load_train_test_splits(self,
                               allowed_categories: Iterable[int] = None,
                               train_size: float = 0.8
                               )  -> Tuple[Dict[int, List[LoCEMultilayerStorage]], Dict[int, List[LoCEMultilayerStorage]]]:
        """
        Get 2 dictionaries of LoCEMultilayerStorages lists per category, where one dict is 'train' part, second one is 'test' part

        Kwargs:
            allowed_categories (Iterable[int] = None): load only LoCE storage of allowed categories ids, None to load all
            test_size (float = 0.2): size of 'test' split

        Returns:
            loce_storages_dict_train (Dict[int, List[LoCEMultilayerStorage]]) 'train' lists of LoCE storages per category
            loce_storages_dict_test (Dict[int, List[LoCEMultilayerStorage]]) 'test' lists of LoCE storages per category
        """
        loce_storages_dict = self.load(allowed_categories)

        # init output dicts
        loce_storages_dict_train = dict()
        loce_storages_dict_test = dict()

        # split each category separately
        for k in sorted(loce_storages_dict.keys()):
            loce_storages_k = loce_storages_dict[k]
            split_idx = int(len(loce_storages_k) * train_size)
            loce_storages_dict_train[k] = loce_storages_k[:split_idx]
            loce_storages_dict_test[k] = loce_storages_k[split_idx:]

        return loce_storages_dict_train, loce_storages_dict_test


class LoCEMultilayerStorageFilesLoader(LoCEMultilayerStorageDirectoryLoader):

    def __init__(self,
                 working_directory: str,
                 files_to_load: Iterable[str],
                 seed: int = None,
                 min_seg_area: float = 0.0,
                 max_seg_area: float = 1.0
                 ) -> None:
        """
        Args:
            working_directory (str): working directory for input-output of LoCEMultilayerStorages
            files_to_load (Iterable[str]): LoCEMultilayerStorages files to load

        Kwargs:
            seed (int = None) seed for data sampling
            min_seg_area (float): minimal allowed segmentation area, LoCEs with smaller segmentation will be ignored
            max_seg_area (float): maximal allowed segmentation area, LoCEs with larger segmentation will be ignored
        """
        self.working_directory = working_directory
        self.seed = seed
        random.seed(seed)

        self.min_seg_area = min_seg_area
        self.max_seg_area = max_seg_area

        self.pkl_file_names = files_to_load

        self._filter_files_by_segmentation_size()


class LoCEMultilayerStorageMultiDirectoryLoader(LoCEMultilayerStorageDirectoryLoader):

    def __init__(self,
                 working_directories: List[str],
                 seed: int = None,
                 min_seg_area: float = 0.0,
                 max_seg_area: float = 1.0
                 ) -> None:
        """
        Args:
            working_directories (List[str]): working directory for input-output of LoCEMultilayerStorages

        Kwargs:
            seed (int = None) seed for data sampling
            min_seg_area (float): minimal allowed segmentation area, LoCEs with smaller segmentation will be ignored
            max_seg_area (float): maximal allowed segmentation area, LoCEs with larger segmentation will be ignored
        """
        self.working_directories = working_directories
        self.seed = seed
        random.seed(seed)

        self.min_seg_area = min_seg_area
        self.max_seg_area = max_seg_area

        self.pkl_file_names = self._select_pkl_files()

        self._filter_files_by_segmentation_size()

    def _select_pkl_files(self) -> List[str]:
        """
        Find file names of pickles with LoCEs

        Returns:
            selected_files (List[str]): filtered files by extension (only .pkl)
        """
        selected_files = []

        for wd in self.working_directories:
            all_files_in_dir = sorted(os.listdir(wd))            

            for fn in all_files_in_dir:

                if fn.split('.')[-1] == 'pkl':
                    selected_files.append(os.path.join(wd, fn))
                else:
                    continue

        return selected_files
    

class LoCEMultilayerStorageStats(LoCEMultilayerStorageDirectoryLoader):
    
    def __init__(self,
                 working_directory: str,
                 seed: int = None,
                 min_seg_area: float = 0.0,
                 max_seg_area: float = 1.0) -> None:
        super().__init__(working_directory, seed, min_seg_area, max_seg_area)

        self.storages = None

    
    def print_meta_info(self, categories: List[int]) -> None:
        self.storages = self.load(categories)

        storages_list = [item for sublist in self.storages.values() for item in sublist]

        n_segmentations = len(storages_list)

        unique_images = set([s.image_path for s in storages_list])
        n_images = len(unique_images)

        print(f"Images: {n_images}, Segmentations: {n_segmentations}")

    
    def _get_stats_one(self, storages: List[LoCEMultilayerStorage]) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, np.ndarray, np.ndarray]], List[str]]:

        layers = list(storages[0].loce_storage.keys())
        loss_dict = {layer: [] for layer in layers}

        for s in storages:
            [loss_dict[layer].append(s.loce_storage[layer].loss) for layer in loss_dict.keys()]

        loss_dict = {layer: np.array(losses) for layer, losses in loss_dict.items()}
        other_stats = {layer: (int((losses == 0).sum()) / len(losses), losses.min(), losses.max()) for layer, losses in loss_dict.items()}

        return loss_dict, other_stats, layers


    def stats_one_category(self,
                           category: int
                           ) -> None:
        
        self.storages = self.load([category])
        
        storages_temp = self.storages[category]

        if len(storages_temp) > 0:

            loss_dict, other_stats, layers = self._get_stats_one(storages_temp)

            print(f"{category} IoUs 0.5:", '; '.join([f"{shorten_layer_name(layer)}: {loss.mean():.2f}±{loss.std():.2f}" for layer, loss in loss_dict.items()]))

            print(f"zero/min/max", '; '.join([f"{shorten_layer_name(layer)}: {l_zero:.2f}/{l_min:.2f}/{l_max:.2f}" for layer, (l_zero, l_min, l_max) in other_stats.items()]))
            
        else:
            print("No storages of given category")

    def stats_numeric(self, categories: List[int]) -> Tuple[Dict[str, int], Dict[str, np.array], Dict[str, Tuple[int, float, float]], List[str]]:
        self.storages = self.load(categories)

        self.storages = {k: v for k, v in self.storages.items() if len(v) > 0}

        categories = [c for c in categories if c in self.storages]

        iou_dict = dict()
        zero_max_min_dict = dict()

        n_samples_dict = dict()

        for c in categories:
            storages_temp = self.storages[c]
            loss_dict, zero_max_min, layers = self._get_stats_one(storages_temp)
            iou_dict[c] = loss_dict
            zero_max_min_dict[c] = zero_max_min
            n_samples_dict[c] = len(storages_temp)
        
        return n_samples_dict, iou_dict, zero_max_min_dict, layers
    
    def stats_numeric_aggregated(self, categories: List[int]) -> Tuple[int, np.array, Tuple[int, float, float], List[str]]:
        self.storages = self.load(categories)

        self.storages = {k: v for k, v in self.storages.items() if len(v) > 0}

        categories = [c for c in categories if c in self.storages]

        iou_dict = dict()
        zero_max_min_dict = dict()

        n_samples_dict = dict()

        for c in categories:
            storages_temp = self.storages[c]
            loss_dict, zero_max_min, layers = self._get_stats_one(storages_temp)
            iou_dict[c] = loss_dict
            zero_max_min_dict[c] = zero_max_min
            n_samples_dict[c] = len(storages_temp)

        n_samples = sum([v for v in n_samples_dict.values()])
        iou_aggregated = dict()
        zero_max_min_aggregated = dict()

        for l in layers:
            iou_aggregated[l] = np.hstack([iou_dict[c][l] for c in categories])
            zero_max_min_aggregated[l] = [sum([zero_max_min_dict[c][l][i]*n_samples_dict[c] for c in categories]) / n_samples for i in range(3)]
        
        return n_samples, iou_aggregated, zero_max_min_aggregated, layers
    
    @staticmethod
    def get_header_str(layers: Iterable[str],
                       idx_width: int,
                       col_width: int
                       ) -> None:
        hdr = 'Category'.ljust(idx_width) + ' & ' + '  &  '.join([shorten_layer_name(l).center(col_width) for l in layers])
        return hdr

    @staticmethod
    def get_row_str(n_samples_dict,
                    iou_dict,
                    zero_max_min_dict,
                    categ_idx: int,
                    layers: Iterable[str],
                    idx_width: int,
                    col_width: int,
                    categories_dict: Dict[int, str]
                    ) -> None:
        
            row = f"{categories_dict[categ_idx]} ({n_samples_dict[categ_idx]:3d})".rjust(idx_width)  + ' & ' +  '  &  '.join([f"{zero_max_min_dict[categ_idx][l][0]:.2f}    {iou_dict[categ_idx][l].mean():.2f}".ljust(col_width) for l in layers])

            return row
    
    @staticmethod
    def get_row_str_zeros(n_samples_dict,
                          iou_dict,
                          zero_max_min_dict,
                          categ_idx: int,
                          layers: Iterable[str],
                          idx_width: int,
                          col_width: int,
                          categories_dict: Dict[int, str]
                          ) -> None:
        
            row = f"{categories_dict[categ_idx]} ({n_samples_dict[categ_idx]:3d})".rjust(idx_width)  + ' & ' +  ' & '.join([f"{zero_max_min_dict[categ_idx][l][0]:.2f}".ljust(col_width) for l in layers])

            return row
    
    @staticmethod
    def get_row_str_iou(n_samples_dict,
                        iou_dict,
                        zero_max_min_dict,
                        categ_idx: int,
                        layers: Iterable[str],
                        idx_width: int,
                        col_width: int,
                        categories_dict: Dict[int, str]
                        ) -> None:
        
            row = f"{categories_dict[categ_idx]} ({n_samples_dict[categ_idx]:3d})".rjust(idx_width)  + ' & ' +  ' & '.join([f"{iou_dict[categ_idx][l].mean():.2f}±{iou_dict[categ_idx][l].std():.2f}".ljust(col_width) for l in layers])

            return row
    
    @staticmethod
    def get_overall_row_str(n_samples_dict,
                            iou_dict,
                            zero_max_min_dict,
                            categories: Iterable[int],
                            layers: Iterable[str],
                            idx_width: int,
                            col_width: int,
                            row_tag: str = "overall"
                            ) -> None:
            
            n_samples_total = sum([n_samples_dict[c] for c in categories])

            zero_iou_dict = dict()

            for l in layers:
                zero_max_min_total = sum([zero_max_min_dict[c][l][0]*n_samples_dict[c] for c in categories]) / n_samples_total  # weighted average
                iou_total =  np.hstack([iou_dict[c][l] for c in categories]).mean()
                zero_iou_dict[l] = (zero_max_min_total, iou_total)
        
            row = f"{row_tag} ({n_samples_total:3d})".rjust(idx_width)  + ' & ' +  '  &  '.join([f"{zero_iou_dict[l][0]:.2f}    {zero_iou_dict[l][1]:.2f}".ljust(col_width) for l in layers])

            return row
    
    @staticmethod
    def get_overall_row_str_zeros(n_samples_dict,
                                  iou_dict,
                                  zero_max_min_dict,
                                  categories: Iterable[int],
                                  layers: Iterable[str],
                                  idx_width: int,
                                  col_width: int,
                                  row_tag: str = "overall"
                                  ) -> None:
            
            n_samples_total = sum([n_samples_dict[c] for c in categories])

            zero_iou_dict = dict()

            for l in layers:
                zero_max_min_total = sum([zero_max_min_dict[c][l][0]*n_samples_dict[c] for c in categories]) / n_samples_total  # weighted average
                iou_total =  np.hstack([iou_dict[c][l] for c in categories]).mean()
                zero_iou_dict[l] = (zero_max_min_total, iou_total)
        
            row = f"{row_tag} ({n_samples_total:3d})".rjust(idx_width)  + ' & ' +  ' & '.join([f"{zero_iou_dict[l][0]:.2f}".ljust(col_width) for l in layers])

            return row
    
    @staticmethod
    def get_overall_row_str_iou(n_samples_dict,
                                iou_dict,
                                zero_max_min_dict,
                                categories: Iterable[int],
                                layers: Iterable[str],
                                idx_width: int,
                                col_width: int,
                                row_tag: str = "overall"
                                ) -> None:
            
            n_samples_total = sum([n_samples_dict[c] for c in categories])

            zero_iou_dict = dict()

            for l in layers:
                zero_max_min_total = sum([zero_max_min_dict[c][l][0]*n_samples_dict[c] for c in categories]) / n_samples_total  # weighted average
                iou_total =  np.hstack([iou_dict[c][l] for c in categories])
                zero_iou_dict[l] = (zero_max_min_total, iou_total)
        
            row = f"{row_tag} ({n_samples_total:3d})".rjust(idx_width)  + ' & ' +  ' & '.join([f"{zero_iou_dict[l][1].mean():.2f}±{zero_iou_dict[l][1].std():.2f}".ljust(col_width) for l in layers])

            return row

    def print_stats(self,
                    categories: List[int],
                    categories_dict: Dict[int, str] = MSCOCO_CATEGORIES
                    ) -> None:

        n_samples_dict, iou_dict, zero_max_min_dict, layers = self.stats_numeric(categories)
        
        idx_width = 16
        col_width = 12

        hdr = self.get_header_str(layers, idx_width, col_width)
        print(hdr)

        #hdr = 'Category'.ljust(idx_width) + ' & ' + ' & '.join([l[-col_width:].ljust(col_width) for l in layers])
        #print(hdr)

        for categ_idx in categories:
            row = self.get_row_str(n_samples_dict, iou_dict, zero_max_min_dict, categ_idx, layers, idx_width, col_width, categories_dict)
            print(row)
            #row = l[-l_width:] + ' & ' +  ' & '.join([f"{loss_dicts[c][l].mean():.2f}±{loss_dicts[c][l].std():.2f}".ljust(width) for c in categories])
            #row = f"{MSCOCO_CATEGORIES[c]} ({n_samples_dict[c]:3d})".rjust(idx_width)  + ' & ' +  ' & '.join([f"{zero_max_min_dict[c][l][0]:.2f}  {iou_dict[c][l].mean():.2f}".ljust(col_width) for l in layers])

            #print(row)

        row_overall = self.get_overall_row_str(n_samples_dict, iou_dict, zero_max_min_dict, categories, layers, idx_width, col_width)
        print(row_overall)

        zeros = []
        losses = []

        for l in layers:
            zeros.append(np.array([float(zero_max_min_dict[c][l][0]) for c in categories]).mean())
            losses.append(np.array([float(iou_dict[c][l].mean()) for c in categories]).mean())

        zeros = np.array(zeros)
        losses = np.array(losses)

        rank_zeros = rankdata(zeros, method='min')
        rank_losses = rankdata(-losses, method='min')

        sum_ranks = rank_zeros + rank_losses

        idx_zeros = np.argmin(rank_zeros)
        idx_losses = np.argmin(rank_losses)
        idx_sum = np.argmin(sum_ranks)

        print(f"Best layer (failures): {layers[idx_zeros]}")
        print(f"Best layer (mean IoU): {layers[idx_losses]}")
        print(f"Best layer (overall) : {layers[idx_sum]}")

        """hdr = 'Layer'.ljust(l_width) + ' & ' + ' & '.join([f"{c}".ljust(width) for c in categories])
        print(hdr)

        for l in layers:

            #row = l[-l_width:] + ' & ' +  ' & '.join([f"{loss_dicts[c][l].mean():.2f}±{loss_dicts[c][l].std():.2f}".ljust(width) for c in categories])
            row = l[-l_width:] + ' & ' +  ' & '.join([f"{stats[c][l][0]:.2f}  {loss_dicts[c][l].mean():.2f}".ljust(width) for c in categories])

            print(row)"""
        
    def plot_stats(self,
                   net_tag: str,
                   net_full_name: str,
                   save_dir: str,
                   aggregates: Dict[str, Dict[str, Iterable[int]]]
                   ) -> None:

        _, _, _, layers = self.stats_numeric_aggregated(list(aggregates.values())[0]["categories"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(len(layers)*0.8*2 + 1.5 + 1, 6))

        num_aggregates = len(aggregates)
        bar_width = 0.8 / num_aggregates  # Width of each bar
        bar_positions = np.arange(len(layers))  # Initial positions for bars

        overall_handles = []
        overall_labels = []

        for i, (aggragate_name, aggragate_content) in enumerate(aggregates.items()):
            aggragate_categories = aggragate_content["categories"]
            aggragate_marker = aggragate_content["marker"]
            aggragate_color = aggragate_content["color"]

            n_samples, iou_agg, zero_max_min_agg, layers = self.stats_numeric_aggregated(aggragate_categories)
            
            layers_short = [shorten_layer_name(l) for l in layers]  # layer names shortened
            iou_means = [iou_agg[l].mean() for l in layers]  # means of iou in each layer
            iou_stds = [iou_agg[l].std() for l in layers]  # std of iou in each layer
            zero_vals = [zero_max_min_agg[l][0] for l in layers]  # int values, natural number
            
            # Bar plot for zero_vals on the left subplot
            bars = ax1.bar(bar_positions + i * bar_width, zero_vals, width=bar_width, color=aggragate_color, alpha=1.0, label=aggragate_name)

            # Line plot with std regions for iou_means on the right subplot
            line, = ax2.plot(layers_short, iou_means, marker=aggragate_marker, color=aggragate_color, label=aggragate_name)
            ax2.fill_between(layers_short, 
                            np.array(iou_means) - np.array(iou_stds), 
                            np.array(iou_means) + np.array(iou_stds), 
                            color=aggragate_color, 
                            alpha=0.2)

            # Collect handles and labels for the legend
            overall_handles.append(line)
            overall_labels.append(aggragate_name)

        # Set y-axis limits between 0 and 1
        ax1.set_ylim(0, 1)
        ax2.set_ylim(0, 1)

        # Set labels and titles
        ax1.set_ylabel('Non-converged')
        ax1.set_xticks(bar_positions + bar_width * (num_aggregates - 1) / 2)
        ax1.set_xticklabels(layers_short, rotation=45, ha="right")
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)  # Add horizontal grid lines

        ax2.set_ylabel('IoU Mean')
        ax2.grid(True, linestyle='--', alpha=0.7)  # Add grid to the right plot
        ax2.set_xticks(bar_positions)
        ax2.set_xticklabels(layers_short, rotation=45, ha="right")

        # Add a legend to the figure
        fig.legend(handles=overall_handles, labels=overall_labels, loc='center right', ncol=1, borderaxespad=0.0)

        # Show the plot
        plt.suptitle(net_full_name)
        #plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"optimization_stats_{net_tag}.pdf"))
        plt.close()
        
    def save_stats(self,
                   categories: List[int],
                   net_tag: str,
                   net_full_name: str,
                   file: IO,
                   aggregates: Dict[str, Dict[str, Iterable[int]]],
                   categories_dict: Dict[int, str] = MSCOCO_CATEGORIES
                   ) -> None:

        n_samples_dict, iou_dict, zero_max_min_dict, layers = self.stats_numeric(categories)
        
        idx_width = 16
        col_width = 9

        file.write("\n" + r"\begin{table*}[ht]" + "\n")
        file.write("\t" + r"\centering" + "\n")
        file.write("\t" + r"\fontsize{5pt}{6pt}\selectfont" + "\n")
        file.write("\t" + r"\setlength{\tabcolsep}{2pt} " + "\n")
        file.write("\t" + r"\begin{tabular}{l|" + "|".join(["c"] * len(layers)) + r"}" + "\n")

        hdr = self.get_header_str(layers, idx_width, col_width)
        file.write("\t\t" + hdr + r" \\" + "\n")
        file.write("\t\t" + r"\hline" + "\n")

        #hdr = 'Category'.ljust(idx_width) + ' & ' + ' & '.join([l[-col_width:].ljust(col_width) for l in layers])
        #print(hdr)

        for categ_idx in categories:
            row_zeros = self.get_row_str_zeros(n_samples_dict, iou_dict, zero_max_min_dict, categ_idx, layers, idx_width, col_width, categories_dict)
            row_iou = self.get_row_str_iou(n_samples_dict, iou_dict, zero_max_min_dict, categ_idx, layers, idx_width, col_width, categories_dict)
            row_zeros_ext = "\t\t" + row_zeros + r" \\" + "\n"
            row_iou_ext = "\t\t" + row_iou + r" \\" + "\n"
            file.write(row_zeros_ext)
            file.write(row_iou_ext)
            #row = l[-l_width:] + ' & ' +  ' & '.join([f"{loss_dicts[c][l].mean():.2f}±{loss_dicts[c][l].std():.2f}".ljust(width) for c in categories])
            #row = f"{MSCOCO_CATEGORIES[c]} ({n_samples_dict[c]:3d})".rjust(idx_width)  + ' & ' +  ' & '.join([f"{zero_max_min_dict[c][l][0]:.2f}  {iou_dict[c][l].mean():.2f}".ljust(col_width) for l in layers])

            #print(row)

        file.write("\t\t" + r"\hline" + "\n")

        for aggragate_name, aggragate_content in aggregates.items():
            aggragate_categories = aggragate_content["categories"]
            row_zeros = self.get_overall_row_str_zeros(n_samples_dict, iou_dict, zero_max_min_dict, aggragate_categories, layers, idx_width, col_width, row_tag=aggragate_name)
            row_iou = self.get_overall_row_str_iou(n_samples_dict, iou_dict, zero_max_min_dict, aggragate_categories, layers, idx_width, col_width, row_tag=aggragate_name)
            row_zeros_ext = "\t\t" + row_zeros + r" \\" + "\n"
            row_iou_ext = "\t\t" + row_iou + r" \\" + "\n"
            file.write(row_zeros_ext)
            file.write(row_iou_ext)

        file.write("\t" + r"\end{tabular}")
        file.write("\t" + r"\caption{" + f"LoCE ablation {net_full_name}." r"}" + "\n")
        file.write("\t" + r"\label{tab:stats-" + f"{net_tag}" r"}" + "\n")
        file.write(r"\end{table*}" + "\n" + "\n")


class LoCEMultilayerStorageVisuals(LoCEMultilayerStorageDirectoryLoader):

    def __init__(self,
                 working_directory: str,
                 seed: int = None,
                 min_seg_area: float = 0.0,
                 max_seg_area: float = 1.0) -> None:
        super().__init__(working_directory, seed, min_seg_area, max_seg_area)

        self.storages = self.load()

    def visuals(self,
                categories: List[int],
                save_dir: str,
                model_tag: str,
                image_tile_size: Tuple[int, int] = (180, 135),
                frame_width: int = 5,
                layers: Iterable[str] = None
                ) -> None:

        categories = [c for c in categories if c in self.storages]

        for c in categories:
            storages_temp = self.storages[c]
            # save_storages_as_tiles_together(storages_temp, save_dir, model_tag, c, image_tile_size, frame_width, layers)
            self._save_storages_as_tiles_separately(storages_temp, save_dir, model_tag, c, image_tile_size, frame_width, layers)

    def _save_storages_as_tiles_strip(self,
                                      loce_storages: Iterable[LoCEMultilayerStorage],
                                      img_out_path: str,
                                      model_tag: str,
                                      category_id: int,
                                      layer: str,
                                      storages_tag: str,
                                      image_tile_size: Tuple[int, int],
                                      frame_width: int
                                      ) -> None:

        if storages_tag == "correct":
            frame_color = "white"
        else:
            frame_color = "black"

        save_path = f'{img_out_path}/{model_tag}/{storages_tag}_{category_id}_{layer}.jpg'
        if os.path.exists(save_path):
            return
        
        # evaluation of tile cells
        tile_cells = find_closest_square_rootable_number(len(loce_storages))

        # finalize number of rows and cols
        cols = rows = int(math.sqrt(tile_cells))
        for r in range(rows, 0, -1):
            if (r * cols) >= len(loce_storages):
                rows = r
            else:
                break

        # canvas
        canvas_width = image_tile_size[0] * cols
        canvas_height = image_tile_size[1] * rows * 2
        canvas = Image.new("RGB", (canvas_width, canvas_height), color='white')

        for row in range(rows):
            for col in range(cols):
                # offsets
                x_offset = col * image_tile_size[0]
                y_offset = row * image_tile_size[1] * 2
                y_offset2 = y_offset + image_tile_size[1]

                # id
                temp_idx = row * cols + col

                # leave space empty if no more images
                if temp_idx >= len(loce_storages):
                    continue

                storage = loce_storages[temp_idx] 
            
                # get current image, segmentation
                temp_image_path = storage.image_path
                temp_proj = storage.loce_storage[layer].projection
                temp_loss = storage.loce_storage[layer].loss
                temp_seg = storage.segmentation

                rgb_mask = get_rgb_binary_mask(temp_proj)
                
                img_with_proj = blend_imgs(Image.fromarray(rgb_mask), Image.open(temp_image_path), alpha=0.5)
                #img_with_proj_counturs = Image.fromarray(add_countours_around_mask(np.array(img_with_proj), temp_proj, countour_color=(255, 0, 0), thickness=1))
                img_with_proj_resized = img_with_proj.resize(image_tile_size)

                img_with_seg = blend_imgs(get_colored_mask(temp_seg, mask_value_multiplier=255), Image.open(temp_image_path), alpha=0.66)
                img_with_seg_counturs = Image.fromarray(add_countours_around_mask(np.array(img_with_seg), temp_seg, countour_color=(0, 255, 0), thickness=5))
                img_with_seg_resized = img_with_seg_counturs.resize(image_tile_size)

                # add colored frame
                draw_frame(ImageDraw.Draw(img_with_proj_resized), image_tile_size, frame_color, frame_width)
                canvas.paste(img_with_proj_resized, (x_offset, y_offset))

                # add colored frame
                draw_frame(ImageDraw.Draw(img_with_seg_resized), image_tile_size, frame_color, frame_width)
                canvas.paste(img_with_seg_resized, (x_offset, y_offset2))

        canvas.save(save_path)

    def _save_storages_as_tiles_separately(self,
                                           loce_storages: Iterable[LoCEMultilayerStorage],
                                           img_out_path: str,
                                           model_tag: str,
                                           category_id: int,
                                           image_tile_size: Tuple[int, int],
                                           frame_width: int,
                                           layers: Iterable[str] = None
                                           ) -> None:

        mkdir(f"{img_out_path}/{model_tag}/")

        # layers of storage
        if layers is None:
            layers = list(loce_storages[0].loce_storage.keys())

        for layer in layers:

            storages_correct = []
            storages_incorrect = []

            for storage in loce_storages:
                storage_loss = storage.loce_storage[layer].loss
                if storage_loss == 0:
                    storages_incorrect.append(storage)
                else:
                    storages_correct.append(storage)

            if len(storages_correct) > 0:
                self._save_storages_as_tiles_strip(storages_correct, img_out_path, model_tag, category_id, layer, "correct", image_tile_size, frame_width)
            if len(storages_incorrect) > 0:    
                self._save_storages_as_tiles_strip(storages_incorrect, img_out_path, model_tag, category_id, layer, "incorrect", image_tile_size, frame_width)

    def loce_weights_distribution(self,
                                  categories: List[int],
                                  save_dir: str,
                                  net_tag: str,
                                  layers: Iterable[str] = None,
                                  categories_dict: Dict[int, str] = MSCOCO_CATEGORIES
                                  ) -> None:

        mkdir(save_dir)

        categories = [c for c in categories if c in self.storages]

        if layers is None:            
            layers = self.storages[categories[0]][0].get_storage_layers()
        
        # gather loces per category and accumulated
        loces_dict = {c: {l: [] for l in layers} for c in categories}
        loces_accumulated_dict = {l: [] for l in layers}

        for c in categories:
            for l in layers:
                vects = [s.get_loce(l).loce for s in self.storages[c]]
                loces_dict[c][l].extend(vects)
                loces_accumulated_dict[l].extend(vects)

        # convert gathered to numpy
        loces_dict = {c: {l: np.array(loces_dict[c][l]) for l in layers} for c in categories}
        loces_accumulated_dict = {l: np.array(loces_accumulated_dict[l]) for l in layers}

        n_plots = len(categories) + 1  # all categories separately + accumulated

        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), sharex=True, sharey='row')
    
        layer_handles = []
        layer_labels = []
        
        for col, categ_idx in enumerate(["All Categories"] + categories):
            ax = axes[col]

            if col == 0:
                    
                category_name = categ_idx

                for l in layers:

                    values = zscore(loces_accumulated_dict[l], axis=1).flatten()
                    kdeplot = sns.kdeplot(values, label=l, fill=True, ax=ax)

                # handles for legend
                handle, label = kdeplot.get_legend_handles_labels()
                layer_handles.extend(handle)
                layer_labels.extend(label)

            if col != 0:

                category_name = categories_dict[categ_idx]

                for l in layers:

                    values = zscore(loces_dict[categ_idx][l], axis=1).flatten()
                    kdeplot = sns.kdeplot(values, label=l, fill=True, ax=ax)
                    
            ax.set_title(f'{category_name}')

            ax.set_xlabel('Value')
            if col == 0:
                ax.set_ylabel('Density')
            #ax.set_ylim(0, 1) 
            ax.grid(True)

        fig.suptitle(f"KDE plots for {net_tag.upper()}")
        
        # Add single legend outside the plot
        fig.legend(handles=layer_handles, labels=layer_labels, loc='center left', ncol=1, borderaxespad=0.0)
        
        plt.savefig(os.path.join(save_dir, f"kde_{net_tag}.pdf"))
        plt.close()
