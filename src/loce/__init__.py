'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

from .loce import LoCE, LoCEWithMetaInformation, LoCEMultilayerStorage, LoCEMultilayerClusterInfo
from .loce import LoCEMultilayerStorageSaver
from .loce import LoCEMultilayerStorageDirectoryLoader, LoCEMultilayerStorageFilesLoader, LoCEMultilayerStorageMultiDirectoryLoader
from .loce import LoCEMultilayerStorageStats, LoCEMultilayerStorageVisuals

from .loce_semantic_segmenter import AbstractSemanticSegmenter, MSCOCOSemanticSegmentationLoader, MSCOCOEllipseSegmenter, MSCOCORectangleSegmenter


from .loce_utils import YOLO5_LAYERS, SSD_LAYERS, EFFICIENTNET_LAYERS, EFFICIENTNETV2_LAYERS, SQUEEZENET_LAYERS, MOBILENET_LAYERS
from .loce_utils import DETR_LAYERS, SWIN_LAYERS, VIT_LAYERS
from .loce_utils import MSCOCO_CATEGORIES, MSCOCO_CATEGORY_COLORS, MSCOCO_MARKERS
from .loce_utils import VOC_CATEGORIES, VOC_CATEGORY_COLORS, VOC_MARKERS
from .loce_utils import LoCEActivationsTensorExtractor
from .loce_utils import ImageLoader, ImageWithGaussianNoiseLoader, ImageWithSaltPepperNoiseLoader
from .loce_utils import get_projection
from .loce_utils import binary_to_uint8_image, loce_stats, downscale_numpy_img, blend_imgs, get_colored_mask, combine_masks, add_bboxes, get_rgb_binary_mask, get_colored_mask_alt
from .loce_utils import draw_mscoco_categories_and_colors, draw_voc_categories_and_colors, plot_binary_mask, plot_projection, save_cluster_imgs_as_tiles, save_cluster_proj_as_tiles, save_storages_as_tiles_together, draw_frame, find_closest_square_rootable_number
from .loce_utils import yolo5_propagator_builder, ssd_propagator_builder
from .loce_utils import mobilenet_propagator_builder, efficientnet_propagator_builder, squeezenet_propagator_builder, efficientnetv2_propagator_builder
from .loce_utils import detr_propagator_builder, swin_propagator_builder, vit_propagator_builder
from .loce_utils import shorten_layer_name

from .loce_optimizer import LoCEOptimizationEngineMSCOCO, TorchCustomLoCEBatchOptimizer

from .loce_clusterer import LoCEClusterer, LoCEClustererManyLoaders

from .loce_dimension_reducer import LoCEDimensionReducer

from .loce_experiment_constants import LoCEExperimentConstants

from .loce_retrieval import LoCEMultilayerStorageRetrieval
from .loce_separation import LoCEMultilayerStorageSeparation
