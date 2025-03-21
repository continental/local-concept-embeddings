'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''


class LoCEExperimentConstants:
    #NET_TAGS = ['detr']
    #NET_TAGS = ['detr', 'swin', 'vit']  # 'efficientnetv2', 'squeezenet' - unused (in paper) networks
    NET_TAGS = ['yolo', 'mobilenet', 'efficientnet', 'detr', 'vit', 'swin']  #  'ssd', 'efficientnetv2', 'squeezenet' - unused (in paper) networks

    NET_FULL_NAMES = {
    'yolo': 'YOLOv5',#'YOLOv5s',
    'ssd': 'SSD',
    'efficientnet': 'EfficientNet', #'EffNet-B0',
    'mobilenet': 'MobileNetV3',#'MobNetV3-L',
    'detr': 'DETR',#'DETR-R50',
    'swin': 'SWIN',#'SWIN-T',
    'vit': 'ViT'#'ViT-B-16'
    }

    CLUSTERING_SETTINGS = ['strict', 'relaxed']

    N_SAMPLES_PER_TAG = 100
    DISTANCE = 'cosine'
    METHOD = 'complete'
    
    PURITY = {
        'strict': 0.90,
        'relaxed': 0.80
    }

    SAMPLE_THRESHOLD_COEFFICIENT = {
        'strict': 0.025,
        'relaxed': 0.05
    }

    # car, motorcycle, airplane, cat, horse, elephant
    MSCOCO_CATEGORIES = [3, 4, 5, 17, 19, 22]

    MSCOCO_CATEGORIES_CAR_BUS_TRUCK = [3, 6, 8]

    MSCOCO_CATEGORIES_ALL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

    MSCOCO_CATEGORIES_ALL_CAPY = [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 383870010]

    MSCOCO_CATEGORIES_VEHICLES = [2, 3, 4, 5, 6, 7, 8, 9]
    MSCOCO_CATEGORIES_ANIMALS = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    MSCOCO_CATEGORIES_ANIMALS_CAPY = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 383870010]

    VOC_CATEGORIES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    N_SEEDS = 50
    K_NEIGHBOURS = [3, 5, 7, 9, 11]

    IOU_LAYERS = {
        'yolo': ["6.cv3.conv", "14.conv", "20.cv3.conv"],
        'efficientnet': ["features.5.2", "features.6.2", "features.7.0"],                  
        'mobilenet': ["features.9", "features.11", "features.14"],
        # transformers
        'detr': ["model.backbone.conv_encoder.model.layer3", "model.encoder.layers.1", "model.encoder.layers.5"],
        'swin': ["features.5", "features.7"],
        'vit': ["encoder.layers.encoder_layer_3", "encoder.layers.encoder_layer_7", "encoder.layers.encoder_layer_11"]
    }

    SUBCONCEPT_LAYERS = {
        'yolo': ["6.cv3.conv", "14.conv", "20.cv3.conv"],
        'efficientnet': ["features.5.2", "features.6.2", "features.7.0"],                  
        'mobilenet': ["features.9", "features.11", "features.14"],
        # transformers
        'detr': ["model.backbone.conv_encoder.model.layer3", "model.encoder.layers.1", "model.encoder.layers.5"],
        'swin': ["features.5", "features.7"],
        'vit': ["encoder.layers.encoder_layer_3", "encoder.layers.encoder_layer_7", "encoder.layers.encoder_layer_11"]
    }

    MAP_LAYERS = {
        'yolo': ["6.cv3.conv", "14.conv", "20.cv3.conv"],
        'efficientnet': ["features.5.2", "features.6.2", "features.7.0"],                  
        'mobilenet': ["features.9", "features.11", "features.14"],
        # transformers
        'detr': ["model.backbone.conv_encoder.model.layer3", "model.encoder.layers.1", "model.encoder.layers.5"],
        'swin': ["features.3", "features.5", "features.7"],
        'vit': ["encoder.layers.encoder_layer_3", "encoder.layers.encoder_layer_7", "encoder.layers.encoder_layer_11"]
    }

    IMG_SAVE_LAYERS = {
        'yolo': ["4.cv3.conv", "6.cv3.conv", "10.conv", "14.conv", "17.cv3.conv", "20.cv3.conv", "23.cv3.conv"],
        'efficientnet': ["features.4.2", "features.5.2", "features.6.2", "features.7.0"],                  
        'mobilenet': ["features.9", "features.11", "features.14", "features.15"],
        # transformers
        'detr': ["model.backbone.conv_encoder.model.layer3", "model.input_projection", "model.encoder.layers.1", "model.encoder.layers.3",  "model.encoder.layers.5"],
        'swin': ["features.5", "features.7"],
        'vit': ["conv_proj", "encoder.layers.encoder_layer_3", "encoder.layers.encoder_layer_7", "encoder.layers.encoder_layer_11"]
    }

    LoCE_DIR = {
        # CNNs
        'yolo': r"./experiment_outputs/optimized_loces_coco/loce_original_yolo",
        'ssd': r"./experiment_outputs/optimized_loces_coco/loce_original_ssd",
        'efficientnet': r"./experiment_outputs/optimized_loces_coco/loce_original_efficientnet",
        'efficientnetv2': r"./experiment_outputs/optimized_loces_coco/loce_original_efficientnetv2",
        'mobilenet': r"./experiment_outputs/optimized_loces_coco/loce_original_mobilenet",
        'squeezenet': r"./experiment_outputs/optimized_loces_coco/loce_original_squeezenet",
        # transformers
        'detr': r"./experiment_outputs/optimized_loces_coco/loce_original_detr",
        'swin': r"./experiment_outputs/optimized_loces_coco/loce_original_swin",
        'vit': r"./experiment_outputs/optimized_loces_coco/loce_original_vit"
    }

    LoCE_DIR_VOC = {
        # CNNs
        'yolo': r"./experiment_outputs/optimized_loces_voc/loce_original_yolo",
        'ssd': r"./experiment_outputs/optimized_loces_voc/loce_original_ssd",
        'efficientnet': r"./experiment_outputs/optimized_loces_voc/loce_original_efficientnet",
        'efficientnetv2': r"./experiment_outputs/optimized_loces_voc/loce_original_efficientnetv2",
        'mobilenet': r"./experiment_outputs/optimized_loces_voc/loce_original_mobilenet",
        'squeezenet': r"./experiment_outputs/optimized_loces_voc/loce_original_squeezenet",
        # transformers
        'detr': r"./experiment_outputs/optimized_loces_voc/loce_original_detr",
        'swin': r"./experiment_outputs/optimized_loces_voc/loce_original_swin",
        'vit': r"./experiment_outputs/optimized_loces_voc/loce_original_vit"
    }

    MLLoCE_LAYER_SETS_OLD = {
        # CNNs
        'yolo': [['10.conv'], ['10.conv', '20.cv3.conv'], ['10.conv', '17.cv3.conv', '20.cv3.conv']],
        'ssd': [['backbone.features.21', 'backbone.extra.0.5', 'backbone.extra.1.0']],
        'efficientnet': [['features.5.0', 'features.6.0', 'features.7.0']],
        'efficientnetv2': [['features.4.1', 'features.5.3', 'features.6.14']],                         
        'mobilenet': [['features.12', 'features.14', 'features.15']],
        'squeezenet': [['features.6.expand3x3', 'features.11.expand3x3', 'features.12.expand3x3']],
        # transformers
        'detr': [["model.backbone.conv_encoder.model.layer3", "model.input_projection", "model.encoder.layers.5"]],
        'swin': [["features.3", "features.5", "features.7"]],
        'vit': [["encoder.layers.encoder_layer_5", "encoder.layers.encoder_layer_8", "encoder.layers.encoder_layer_11"]]
    }

    # layers for the fair comparison, last (encoder) layers of networks
    MLLoCE_LAYER_SETS = {
        # CNNs
        'yolo': [['10.conv']],
        'ssd': [['backbone.features.19']],
        'efficientnet': [['features.7.0']],
        'efficientnetv2': [['features.6.14']],                         
        'mobilenet': [['features.15']],
        'squeezenet': [['features.12.expand3x3']],
        # transformers
        'detr': [["model.input_projection"], ["model.encoder.layers.1"], ["model.encoder.layers.2"], ["model.encoder.layers.3"], ["model.encoder.layers.4"], ["model.encoder.layers.5"]],
        'swin': [["features.7"]],
        'vit': [["encoder.layers.encoder_layer_11"]]
    }

    CLUSTER_IMGS_FLAG = {  # works only for ODs, because also build bboxes, code cna be fixed - need to remove/comment 2 lines, but who cares
        # CNNs
        'yolo': True,
        'ssd': True,
        'efficientnet': False,
        'efficientnetv2': False,
        'mobilenet': False,
        'squeezenet': False,
        # transformers
        'detr': False, # OD, but lazy to rewrite
        'swin': False,
        'vit': False
    }

    OPTIMIZATION_TYPES = {

    "mse": (1.0, 0.0, 0.0, "downscale"),
    "mse_reg": (1.0, 0.0, 1.0, "downscale"),
    "mse_reg01": (1.0, 0.0, 0.1, "downscale"),
    "mse_reg001": (1.0, 0.0, 0.01, "downscale"),

    "dice": (0.0, 1.0, 0.0, "downscale"),
    "dice_reg": (0.0, 1.0, 1.0, "downscale"),
    "dice_reg01": (0.0, 1.0, 0.1, "downscale"),
    "dice_reg001": (0.0, 1.0, 0.01, "downscale"),
    
    "mse_dice": (1.0, 1.0, 0.0, "downscale"),
    "mse_dice_reg": (1.0, 1.0, 1.0, "downscale"),
    "mse_dice_reg01": (1.0, 1.0, 0.1, "downscale"),
    "mse_dice_reg001": (1.0, 1.0, 0.01, "downscale"),

    "mse_reg01up": (1.0, 0.0, 0.1, "upscale"),
    "dice_reg01up": (0.0, 1.0, 0.1, "upscale"),
    "mse_dice_reg01up": (1.0, 1.0, 0.1, "upscale"),
    }

    ABLATION_LAYERS = {
        'yolo': ['6.cv3.conv',
                 '8.cv3.conv',
                 '9.cv2.conv',
                 '12',
                 '16',
                 '17.cv3.conv',
                 '20.cv3.conv',
                 '23.cv3.conv'],
        'efficientnet': ['features.4.2',
                         'features.5.0',
                         'features.5.1',
                         'features.5.2',
                         'features.6.0',
                         'features.6.1',
                         'features.6.2',
                         'features.7.0'],                  
        'mobilenet': ['features.9',
                      'features.10',
                      'features.11',
                      'features.12',
                      'features.13',
                      'features.14',
                      'features.15'],
        # transformers
        'detr': [
            "model.backbone.conv_encoder.model.layer3",
            "model.input_projection",
            "model.encoder.layers.0",
            "model.encoder.layers.1",
            "model.encoder.layers.2",
            "model.encoder.layers.3",
            "model.encoder.layers.4",
            "model.encoder.layers.5"],
        'swin': ["features.0",
                 "features.1",
                 "features.3",
                 "features.5",
                 "features.7"],
        'vit': [
            "encoder.layers.encoder_layer_1",
            "encoder.layers.encoder_layer_3",
            "encoder.layers.encoder_layer_5",
            "encoder.layers.encoder_layer_7",
            "encoder.layers.encoder_layer_9",
            "encoder.layers.encoder_layer_11"]
    }

