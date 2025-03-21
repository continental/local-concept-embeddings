'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

import os
from PIL import Image, ImageDraw
import numpy as np
from hooks import Propagator, PropagatorTorchClassifier, PropagatorTorchSSD, PropagatorUltralyticsYOLOv5Old, PropagatorHuggingFaceDETR
from xai_utils.files import mkdir, add_countours_around_mask
import torch
import math
from torchvision.models.detection.ssd import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.efficientnet import efficientnet_v2_s, EfficientNet_V2_S_Weights, efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models.mobilenet import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.squeezenet import squeezenet1_1, SqueezeNet1_1_Weights
from torchvision.models.swin_transformer import swin_t, Swin_T_Weights
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from typing import List, Dict, Tuple, Iterable, Any
import matplotlib.pyplot as plt
from matplotlib import colors
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers.image_processing_utils import BaseImageProcessor


MSCOCO_CATEGORY_COLORS = {
    # person
    1: 'deepskyblue',
    # vehicles
    2: 'yellowgreen',
    3: 'firebrick',
    4: 'green',
    5: 'mediumblue',
    6: 'darkorange',
    7: 'darkmagenta',
    8: 'dodgerblue',
    9: 'mediumseagreen',
    # animals
    16: 'violet',
    17: 'darkviolet',
    18: 'darkcyan',
    19: 'deeppink',
    20: 'peru',
    21: 'turquoise',
    22: 'gold',
    23: 'olive',
    24: 'navy',
    25: 'tomato',
    383870010: "red"
    }

MSCOCO_CATEGORIES = {
    # person
    1: 'person',
    # vehicles
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    # animals
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    383870010: "capybara"
}

MSCOCO_MARKERS = {
    # person
    1: '^',
    # vehicles
    2: 'o',
    3: 'o',
    4: 'o',
    5: 'o',
    6: 'o',
    7: 'o',
    8: 'o',
    9: 'o',
    # animals
    16: 's',
    17: 's',
    18: 's',
    19: 's',
    20: 's',
    21: 's',
    22: 's',
    23: 's',
    24: 's',
    25: 's',
    383870010: "X"
}


VOC_CATEGORIES = {
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}


VOC_CATEGORY_COLORS = {
    # preserved from MS COCO where applicable
    1: 'mediumblue',     # aeroplane -> airplane
    2: 'yellowgreen',    # bicycle
    3: 'violet',         # bird
    4: 'mediumseagreen', # boat
    5: 'saddlebrown',    # bottle
    6: 'darkorange',     # bus
    7: 'firebrick',      # car
    8: 'darkviolet',     # cat
    9: 'royalblue',      # chair (new assignment)
    10: 'turquoise',     # cow
    11: 'gold',          # diningtable (new assignment)
    12: 'darkcyan',      # dog
    13: 'deeppink',      # horse
    14: 'green',         # motorbike -> motorcycle
    15: 'deepskyblue',   # person
    16: 'olive',         # pottedplant (new assignment)
    17: 'peru',          # sheep
    18: 'tomato',        # sofa (new assignment)
    19: 'darkmagenta',   # train
    20: 'navy'           # tvmonitor (new assignment)
}

VOC_MARKERS = {
    # preserved from MS COCO where applicable
    1: 'o',  # aeroplane -> airplane
    2: 'o',  # bicycle
    3: 's',  # bird
    4: 'o',  # boat
    5: 'D',  # bottle (new assignment)
    6: 'o',  # bus
    7: 'o',  # car
    8: 's',  # cat
    9: 'D',  # chair (new assignment)
    10: 's', # cow
    11: 'D', # diningtable (new assignment)
    12: 's', # dog
    13: 's', # horse
    14: 'o', # motorbike -> motorcycle
    15: '^', # person
    16: 'D', # pottedplant (new assignment)
    17: 's', # sheep
    18: 'D', # sofa (new assignment)
    19: 'o', # train
    20: 'D'  # tvmonitor (new assignment)
}




YOLO5_LAYERS = ['4.cv3.conv',
                '5.conv',
                '6.cv3.conv',
                '7.conv',
                '8.cv3.conv',
                '9.cv2.conv',
                '10.conv',
                '12',
                '14.conv',
                '16',
                '17.cv3.conv',
                '18.conv',
                '19',
                '20.cv3.conv',
                '21.conv',
                '22',
                '23.cv3.conv']

SSD_LAYERS = ['backbone.features.19',
              'backbone.features.21',
              'backbone.extra.0.1',
              'backbone.extra.0.3',
              'backbone.extra.0.5',
              'backbone.extra.1.0',
              'backbone.features',
              'backbone.extra.0',
              'backbone.extra.1',
              'backbone.extra.2',
              'backbone.extra.3',
              'backbone.extra.4']


MOBILENET_LAYERS = ['features.9',
                    'features.10',
                    'features.11',
                    'features.12',
                    'features.13',
                    'features.14',
                    'features.15']


EFFICIENTNET_LAYERS = ['features.4.2',
                       'features.5.0',
                       'features.5.1',
                       'features.5.2',
                       'features.6.0',
                       'features.6.1',
                       'features.6.2',
                       'features.7.0']

EFFICIENTNETV2_LAYERS = ['features.3',
                         'features.4.1',
                         'features.4.5',
                         'features.5.3',
                         'features.6.0',
                         'features.6.4',
                         'features.6.9',
                         'features.6.14']

SQUEEZENET_LAYERS = ['features.6.expand3x3',
                     'features.7.expand3x3',
                     'features.9.expand3x3',
                     'features.10.expand3x3',
                     'features.11.expand3x3',
                     'features.12.expand3x3']


# B: batch
# H*W - 1d feature dimension size obtained from 2d feature maps
# C / d: channels / reduced channels after feature projection
# N - decoder queries dimension

DETR_LAYERS = [
    # output layer dimensions: [B, C, H, W] - regular CNN backbone layers
    # "model.backbone.conv_encoder.model.layer1",
    # "model.backbone.conv_encoder.model.layer2",
    "model.backbone.conv_encoder.model.layer3",
    "model.backbone.conv_encoder.model.layer4",

    # output layer dimensions: [B, d, H, W] - reduced dimensions / channels
    "model.input_projection",

    # output layer dimensions: [B, H*W, d] - encoder dimensionality, mind the axis swap
    "model.encoder.layers.0",
    "model.encoder.layers.1",
    "model.encoder.layers.2",
    "model.encoder.layers.3",
    "model.encoder.layers.4",
    "model.encoder.layers.5",

    # output layer dimensions: [B, N, d] - decoder is unexplainable with LoCEs because there is no reference for segmentation approximation
    # in forward pass use output_attentions=True and visualize decoder cross-attention instead
    # tutorial here: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/DETR_minimal_example.ipynb
    # "model.decoder.layers.0",
    # "model.decoder.layers.1",
    # "model.decoder.layers.2",
    # "model.decoder.layers.3",
    # "model.decoder.layers.4",
    # "model.decoder.layers.5",
]

SWIN_LAYERS = [
    # conv patches
    "features.0",

    # transformer blocks (odd) with PatchMerging (even) inbetween
    "features.1",
    "features.3",
    "features.5",
    "features.7"
]

VIT_LAYERS = [
    # conv patches
    "conv_proj",

    # layers
    "encoder.layers.encoder_layer_1",
    "encoder.layers.encoder_layer_2",
    "encoder.layers.encoder_layer_3",
    "encoder.layers.encoder_layer_4",
    "encoder.layers.encoder_layer_5",
    "encoder.layers.encoder_layer_6",
    "encoder.layers.encoder_layer_7",
    "encoder.layers.encoder_layer_8",
    "encoder.layers.encoder_layer_9",
    "encoder.layers.encoder_layer_10",
    "encoder.layers.encoder_layer_11",
]

EPSILON = 0.000001


def draw_mscoco_categories_and_colors(mscoco_category_ids: Iterable[int]
                                      ) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Draw MS COCO categories and colors for plotting by id

    Args:
        mscoco_category_ids (Iterable[int]): category ids

    Returns:
        id2name (Dict[int, str]): category id to name dict
        id2color (Dict[int, str]): category id to color dict
    """
    id2name = {i: MSCOCO_CATEGORIES[i] for i in mscoco_category_ids}
    id2color = {i: MSCOCO_CATEGORY_COLORS[i] for i in mscoco_category_ids}
    return id2name, id2color


def draw_voc_categories_and_colors(voc_category_ids: Iterable[int]
                                   ) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Draw PASCAL VOC categories and colors for plotting by id

    Args:
        voc_category_ids (Iterable[int]): category ids

    Returns:
        id2name (Dict[int, str]): category id to name dict
        id2color (Dict[int, str]): category id to color dict
    """
    id2name = {i: VOC_CATEGORIES[i] for i in voc_category_ids}
    id2color = {i: VOC_CATEGORY_COLORS[i] for i in voc_category_ids}
    return id2name, id2color


def blend_imgs(img1: Image, img2: Image, alpha: float = 0.5) -> Image:
    """
    Blend two Image instances: aplha * img1 + (1 - alpha) * img2

    Args:
        img1 (Image): image 1
        img2 (Image): image 2

    Kwargs:
        alpha (np.ndarray = 0.5): alpha for blending

    Returns:
        (Image) blended image
    """
    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")
    img2 = img2.resize(img1.size)
    return Image.blend(img1, img2, alpha)


def get_colored_mask(mask: np.ndarray, color_channels: List[int] = [1], mask_value_multiplier: int = 1) -> Image:
    """
    Expand greyscale mask to RGB dimensions.

    Args:
        mask (np.ndarray): greyscale mask

    Kwargs:
        color_channels (List[int] = [1]): channels to fill with mask values, values of other other channels stay equal to 0. default - green mask
        mask_value_multiplier (int = 1): final value multiplier, use 255 if original mask was boolean, otherwise - 1

    Returns:
        (Image) RGB mask
    """
    rgb_img = np.zeros((*mask.shape, 3), dtype=mask.dtype)
    for c in color_channels:
        rgb_img[:, :, c] = mask
    return Image.fromarray(rgb_img.astype(np.uint8) * mask_value_multiplier)


def get_colored_mask_alt(mask: np.ndarray, color_channels: List[int] = [1], mask_value_multipliers: Iterable[int] = [1, 1, 1]) -> Image:
    """
    Expand greyscale mask to RGB dimensions.

    Args:
        mask (np.ndarray): greyscale mask

    Kwargs:
        color_channels (List[int] = [1]): channels to fill with mask values, values of other other channels stay equal to 0. default - green mask
        mask_value_multiplier (int = 1): final value multiplier, use 255 if original mask was boolean, otherwise - 1

    Returns:
        (Image) RGB mask
    """
    rgb_img = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c in color_channels:
        rgb_img[:, :, c] = mask.astype(np.uint8) * mask_value_multipliers[c]
    return Image.fromarray(rgb_img)


def combine_masks(masks: List[np.ndarray]) -> np.ndarray:
    """
    Combine (and rescale) masks to get an averaged mask.

    Args:
        masks (List[np.ndarray]): list of masks (may have different sizes) to combine

    Returns:
        combined_mask (np.ndarray) rescaled and combined mask
    """
    largest_mask_id = np.argmax([m.size for m in masks])

    img_masks = [Image.fromarray(m) for m in masks]
    new_size = img_masks[largest_mask_id].size
    resized_masks = [np.array(i.resize(new_size)) for i in img_masks]

    combined_mask = np.array(resized_masks).mean(axis=0).astype(np.uint8)

    return combined_mask


def add_bboxes(img: Image,
               predictions: np.ndarray,
               predictions_img_tensor_shape: Tuple[int, int] = (640, 480),
               bbox_color: str = 'red'
               ) -> Image:
    """
    Add bboxes from predictions to the image

    Args:
        img (Image): image instance
        predictions (np.ndarray): numpy array with predictions (N, 6), N - number of bboxes, each bbox has (x1, y1, x2, y2, prob, label)
        predictions_img_tensor_shape (Tuple[int, int] = (640, 480)): (W, H) of original tensor image for which predictions were calculated
        bbox_color (str = 'red'): color of bbox
    """
    img_resized = img.resize(predictions_img_tensor_shape)

    draw = ImageDraw.Draw(img_resized)

    for x1, y1, x2, y2, prob, label in predictions:
        draw.rectangle([(x1, y1), (x2, y2)], outline=bbox_color, width=5)

    return img_resized


def binary_to_uint8_image(arr: np.ndarray) -> np.ndarray:
    """
    Convert binary np.ndarray to np.uint8 image array

    Args:
        arr (np.ndarray): array to convert

    Returns:
        (np.ndarray) np.uint8 image array
    """
    return (arr * 255).astype(np.uint8)


def downscale_numpy_img(img: np.ndarray,
                        downscale_factor: float = 5.0
                        ) -> np.ndarray:
    """
    Downscale numpy image array

    Args:
        img (np.ndarray): image

    Kwargs:
        downscale_factor (float = 5.0): downscale factor

    Returns:
        (np.ndarray) downscaled image
    """
    img = Image.fromarray(img)
    return np.array(img.resize((int(c / downscale_factor) for c in img.size)))


def plot_binary_mask(mask: np.ndarray,
                     downscale_factor: float = 5.0
                     ) -> None:
    """
    Plot (downscaled) binary mask

    Args:
        mask (np.ndarray): mask

    Kwargs:
        downscale_factor (float = 5.0): downscale factor
    """
    img_arr = downscale_numpy_img(mask, downscale_factor)
    img = Image.fromarray(img_arr)
    img.show()


def loce_stats(loce: np.ndarray) -> None:
    """
    Print LoCE stats (mean, var, sparsity)

    Args:
        loce (np.ndarray): LoCE
    """
    print('\tmean:', loce.mean())
    print('\tvar:', loce.var())
    print(f'\tsparsity: {(loce == 0).sum()}/{len(loce)}', )


def plot_projection(loce: np.ndarray,
                    acts: np.ndarray,
                    proj_name: str = None
                    ) -> None:
    """
    Plot projection of LoCE and activations

    Args:
        loce (np.ndarray): LoCE
        acts (np.ndarray): activations

    Kwargs:
        proj_name: projection name to print
    """
    if proj_name is not None:
        print(proj_name)
    loce_stats(loce)

    projecion_uint8 = get_projection(loce, acts)
    plot_binary_mask(projecion_uint8, 0.1)


def get_projection(loce: np.ndarray,
                   acts: np.ndarray,
                   downscale_factor: float = None
                   ) -> np.ndarray:
    """
    Get projection of LoCE and activations

    Args:
        loce (np.ndarray): LoCE
        acts (np.ndarray): activations

    Kwargs:
        downscale_factor (float = None): downscale factor

    Returns:
        (np.ndarray) np.uint8 image array
    """
    def sigmoid(z):
        z = np.clip(z, -10., 10.) # to avoid overflow in np.exp()
        return 1/(1 + np.exp(-z))

    loce3d = np.expand_dims(loce, axis=[1, 2])
    projecion = (acts * loce3d).sum(axis=0)
    projecion = sigmoid(projecion)  # normalize_0_to_1(projecion)

    if downscale_factor is not None:
        projecion = downscale_numpy_img(projecion, downscale_factor)

    projecion_uint8 = (projecion * 255).astype(np.uint8)
    return projecion_uint8


def get_rgb_binary_mask(mask: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:

    def rescale_to_range(data, new_min, new_max):
        old_min = np.min(data)
        old_max = np.max(data)

        rescaled_data = ((data - old_min) / (old_max - old_min +
                         EPSILON)) * (new_max - new_min) + new_min
        return rescaled_data

    img = Image.fromarray(mask.astype(np.float32))
    if target_size:
        img = img.resize(target_size)

    img_np = np.array(img)

    img_np = rescale_to_range(img_np, 0, 1)

    # apply colormap
    cmap = plt.get_cmap('bwr')
    img_rgba = cmap(img_np)
    # rgba to rgb
    img_rgb = (img_rgba[:, :, :3] * 255).astype(np.uint8)

    return img_rgb


def yolo5_propagator_builder(layers: List[str] = YOLO5_LAYERS):
    yolo5 = torch.hub.load('ultralytics/yolov5',
                           'yolov5s', skip_validation=True)

    yolo5_prop = PropagatorUltralyticsYOLOv5Old(yolo5, layers)

    return yolo5_prop


def ssd_propagator_builder(layers: List[str] = SSD_LAYERS):
    ssd = ssd300_vgg16(weights=SSD300_VGG16_Weights)

    ssd_prop = PropagatorTorchSSD(ssd, layers)

    return ssd_prop


def mobilenet_propagator_builder(layers: List[str] = MOBILENET_LAYERS):
    mobilenet = mobilenet_v3_large(
        weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

    mobilenet_prop = PropagatorTorchClassifier(mobilenet, layers)

    return mobilenet_prop


def efficientnet_propagator_builder(layers: List[str] = EFFICIENTNET_LAYERS):
    efficientnet = efficientnet_b0(
        weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    efficientnet_prop = PropagatorTorchClassifier(efficientnet, layers)

    return efficientnet_prop


def efficientnetv2_propagator_builder(layers: List[str] = EFFICIENTNETV2_LAYERS):
    efficientnet = efficientnet_v2_s(
        weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    efficientnet_prop = PropagatorTorchClassifier(efficientnet, layers)

    return efficientnet_prop


def squeezenet_propagator_builder(layers: List[str] = SQUEEZENET_LAYERS):
    squeezenet = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)

    squeezenet_prop = PropagatorTorchClassifier(squeezenet, layers)

    return squeezenet_prop


def detr_propagator_builder(layers: List[str] = DETR_LAYERS):

    detr_processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    detr_prop = PropagatorHuggingFaceDETR(model, layers)

    return detr_prop, detr_processor


def swin_propagator_builder(layers: List[str] = SWIN_LAYERS):
    swin = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)

    swin_prop = PropagatorTorchClassifier(swin, layers)

    return swin_prop


def vit_propagator_builder(layers: List[str] = VIT_LAYERS):
    vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    vit_prop = PropagatorTorchClassifier(vit, layers)

    return vit_prop



class LoCEActivationsTensorExtractor:

    def __init__(self,
                 propagator: Propagator,
                 propagator_tag: str,
                 processor: BaseImageProcessor = None
                 ) -> None:
        
        """
        propagator (Propagator): propagator instance
        propagator_tag (str): propogatr tag
        processor (BaseImageProcessor): HuggingFace image processor (if model requires)
        """
        self.propagator = propagator
        self.propagator_tag = propagator_tag
        self.processor = processor

    def get_bchw_acts_preds_dict(self,
                                 image_pil: Image.Image,
                                 ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get activations and predictions for image

        Args:
            image_pil (Image.Image): image instance

        Returns:
            acts (Dict[str, Tensor]) dictionary with Tensor[B,C,H,W] activations
            preds (Tensor) bbox predictions Tensor[N,6]
        """
        # custom method, HuggingFace model, a bit more complex
        if self.propagator_tag == 'detr':
            return self._get_acts_preds_dict_detr(image_pil)
        
        elif self.propagator_tag == 'vit':
            # append layer to get shape from, if not added
            # TODO: unfortunately it will increase memory use for activations in this layer. need to rework later
            aux_layer_added_flag = False
            if 'conv_proj' not in self.propagator.layers:
                self.propagator.set_layers(['conv_proj'] + self.propagator.layers)
                aux_layer_added_flag = True

            # input size is strictly (224, 224)
            image_pil = image_pil.resize((224, 224))
            acts, preds = self._get_acts_preds_dict(image_pil)

            conv_proj_shape = acts['conv_proj'].shape

            if aux_layer_added_flag:
                self.propagator.set_layers(self.propagator.layers[1:])
                acts.pop('conv_proj')

            for l, a in acts.items():
                # 'conv_proj' is CNN layer, other layers need to be processed
                if l != 'conv_proj':
                    a = a[:, 1:, :] # remove learnable class embedding (0th element) [B, H*W+1, C] -> [B, H*W, C]
                    a = a.permute(0, 2, 1) # swap axes [B, H*W, C] -> [B, C, H*W]
                    a = a.view(conv_proj_shape) # convert to 2d [B, C, H*W] -> [B, C, H, W]
                acts[l] = a

            return acts, preds

        elif self.propagator_tag == 'swin':
            acts, preds = self._get_acts_preds_dict(image_pil)
            acts = {l: a.permute(0, 3, 1, 2) for l, a in acts.items()}
            return acts, preds
        
        # other CNN-models, no need to permute or transform dimensions
        else:
            return self._get_acts_preds_dict(image_pil)


    def _get_acts_preds_dict(self,
                             image_pil: Image.Image
                             ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get activations and predictions for image

        Args:
            image_pil (Image.Image): image instance

        Returns:
            acts (Dict[str, Tensor]) dictionary with activations
            preds (Tensor) bbox predictions Tensor[N,6]
        """
        img_np = np.moveaxis(np.array(image_pil).astype(np.float32) / 255., [2], [0])

        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        acts = self.propagator.get_activations(img_tensor)
        preds = self.propagator.get_predictions(img_tensor)
        return acts, preds


    def _get_acts_preds_dict_detr(self,
                                  image_pil: Image.Image
                                  ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get activations and predictions for image

        Args:
            image_pil (Image.Image): image instance

        Returns:
            acts (Dict[str, Tensor]) dictionary with activations
            preds (Tensor) bbox predictions Tensor[N,6]
        """

        encoding = self.processor(image_pil, return_tensors="pt")

        # append layer to get shape from, if not added
        # TODO: unfortunately it will increase memory use for activations in this layer. need to rework later
        aux_layer_added_flag = False
        if 'model.input_projection' not in self.propagator.layers:
            self.propagator.set_layers(['model.input_projection'] + self.propagator.layers)
            aux_layer_added_flag = True

        acts = self.propagator.get_activations(encoding)

        # remembering CNN shape
        input_proj_shape = acts['model.input_projection'].shape

        if aux_layer_added_flag:
            self.propagator.set_layers(self.propagator.layers[1:])
            acts.pop('model.input_projection')

        activations_dict = {}

        # additional processing of activations
        for layer, act_item in acts.items():

            # ensure that representation is a tensor
            if not isinstance(act_item, torch.Tensor):
                act_item = act_item[0]

            # detach
            activations_dict[layer] = act_item.detach().cpu()

        # convert from transformer to CNN dimensions: [B, H*W, d] -> [B, d, H, W]
        for layer, act_item in activations_dict.items():
            # B: batch
            # H*W - 1d feature dimension size obtained from 2d feature maps
            # C / d: channels / reduced channels after feature projection
            if len(act_item.shape) == 3:
                # [B, H*W, d] -> [B, d, H*W]
                act_item_permuted = act_item.permute(0, 2, 1)
                # [B, d, H*W] -> [B, d, H, W]
                activations_dict[layer] = act_item_permuted.view(input_proj_shape)

        # get and process predictions
        preds = self.propagator.get_predictions(encoding)
        processed_preds = self.processor.post_process_object_detection(preds, target_sizes=[image_pil.size], threshold=0.5)

        # additional processing of predictions to fit (N, 6) shape: box, confidence, label
        predictions_list = []
        for pred_dict in processed_preds:
            prediction = torch.cat((pred_dict['boxes'], pred_dict['scores'].unsqueeze(1), pred_dict['labels'].unsqueeze(1)), dim=1)
            predictions_list.append(prediction.detach().cpu())

        return activations_dict, predictions_list


class ImageLoader:

    def __init__(self,
                 img_shape: Tuple[int, int] = (640, 480)
                 ) -> None:
        self.img_shape = img_shape

    def load_pil_img(self,
                     img_folder: str,
                     img_name: str
                     ) -> Image.Image:
        img_pil = Image.open(os.path.join(img_folder, img_name)).convert('RGB').resize(self.img_shape)
        return img_pil


class ImageWithGaussianNoiseLoader(ImageLoader):
    def __init__(self,
                 img_shape: Tuple[int, int] = (640, 480),
                 noise_level_percent: float = 5.0,  # Noise level as a percentage
                 seed: int = None  # Optional seed for reproducibility
                 ) -> None:
        super().__init__(img_shape)
        self.noise_std = (noise_level_percent / 100.0) * 255  # Convert percentage to std dev
        self.seed = seed

    def load_pil_img(self,
                     img_folder: str,
                     img_name: str
                     ) -> Image.Image:
        # Load and preprocess the image
        img_pil = Image.open(os.path.join(img_folder, img_name)).convert('RGB').resize(self.img_shape)
        img_np = np.array(img_pil)

        # Set the seed for reproducibility
        if self.seed is not None:
            np.random.seed(self.seed)

        # Generate Gaussian noise
        gaussian_noise = np.random.normal(0, self.noise_std, img_np.shape)

        # Add noise to the image
        img_noisy = img_np + gaussian_noise

        # Clip values to be in valid range (0–255) and convert back to uint8
        img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)

        # Convert back to a PIL image and return
        return Image.fromarray(img_noisy)


class ImageWithSaltPepperNoiseLoader(ImageLoader):
    def __init__(self,
                 img_shape: Tuple[int, int] = (640, 480),
                 noise_level_percent: float = 5.0,  # Noise level as a percentage of total pixels
                 seed: int = None  # Optional seed for reproducibility
                 ) -> None:
        super().__init__(img_shape)
        self.noise_level = noise_level_percent / 100.0  # Convert percentage to fraction
        self.seed = seed

    def load_pil_img(self,
                     img_folder: str,
                     img_name: str
                     ) -> Image.Image:
        # Load and preprocess the image
        img_pil = Image.open(os.path.join(img_folder, img_name)).convert('RGB').resize(self.img_shape)
        img_np = np.array(img_pil)

        # Set the seed for reproducibility
        if self.seed is not None:
            np.random.seed(self.seed)

        # Generate salt-and-pepper noise
        num_pixels = img_np.size  # Total number of pixels (width * height * channels)
        num_salt = int(self.noise_level * num_pixels / 2)  # Half for salt
        num_pepper = int(self.noise_level * num_pixels / 2)  # Half for pepper

        # Get random coordinates for salt
        salt_coords = [np.random.randint(0, i, num_salt) for i in img_np.shape]
        # Get random coordinates for pepper
        pepper_coords = [np.random.randint(0, i, num_pepper) for i in img_np.shape]

        # Apply salt (set to 255) and pepper (set to 0)
        img_np[salt_coords[0], salt_coords[1], :] = 255  # Salt (white)
        img_np[pepper_coords[0], pepper_coords[1], :] = 0  # Pepper (black)

        # Convert back to a PIL image and return
        return Image.fromarray(img_np)



def find_closest_square_rootable_number(x):
    y = x
    while True:
        sqrt_y = math.sqrt(y)
        if sqrt_y.is_integer():
            return y
        y += 1


def save_cluster_imgs_as_tiles(clustered_loce_storages: Iterable[Iterable[Any]],
                               selected_tag_id_names: Dict[int, str],
                               selected_tag_ids_and_colors: Dict[int, str],
                               img_out_path: str,
                               image_tile_size: Tuple[int, int] = (256, 192),
                               model_categories: Dict[int, str] = None,
                               frame_width: int = 5,
                               ) -> None:
    """
    Tile images from cluster and save as '{img_out_path}/cluster_{cluster_number}.jpg'

    Args:
        clustered_loce_storages (List[List[LoCEMultilayerStorage]]): list of lists of LoCEMultilayerStorage, where external list is clusters, internal lists are leaf LoCEMultilayerStorage
        selected_tag_ids_and_colors (Dict[int, str]): categories (tag_ids) to perform clustering for and corresponding colors, e.g., {3: 'red', 4: 'blue'}
        selected_tag_ids_and_colors (Dict[int, str]): categories (tag_ids) to perform clustering for and corresponding names, e.g., {3: 'car', 4: 'motorcycle'}
        img_out_path (str): output directory for tiled images

    Kwargs:
        image_tile_size (Tuple[int, int]): size of each tile
        model_categories (str = None): different models may have different id-label mappings, needed for correct visualization of bboxes
    """

    mkdir(img_out_path)

    # number of images per cluster
    cluster_counts = [len(cl) for cl in clustered_loce_storages]

    # evaluation of tile cells
    tile_cells = [find_closest_square_rootable_number(
        i) for i in cluster_counts]

    # for each cluster:
    for cluster_number, (loce_cluster, tc) in enumerate(zip(clustered_loce_storages, tile_cells)):

        # finalize number of rows and cols
        cols = rows = int(math.sqrt(tc))
        for r in range(rows, 0, -1):
            if (r * cols) >= len(loce_cluster):
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
                if temp_idx >= len(loce_cluster):
                    continue

                temp_loce = loce_cluster[temp_idx]

                # get current image, segmentation and color
                temp_category = temp_loce.segmentation_category_id
                temp_category_name = selected_tag_id_names[temp_category]
                # fix mapping of model categories and MS COCO original categories
                model_category = temp_category
                if model_categories is not None:
                    for mc_idx, mc_name in model_categories.items():
                        if mc_name == temp_category_name:
                            model_category = mc_idx
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

        canvas.save(f'{img_out_path}/cluster_{cluster_number}.jpg')


def save_cluster_proj_as_tiles(clustered_loce_storages: Iterable[Iterable[Any]],
                               selected_tag_id_names: Dict[int, str],
                               selected_tag_ids_and_colors: Dict[int, str],
                               img_out_path: str,
                               image_tile_size: Tuple[int, int] = (256, 192),
                               model_categories: Dict[int, str] = None,
                               layers: Iterable[str] = None
                               ) -> None:
    """
    Tile images from cluster and save as '{img_out_path}/cluster_{cluster_number}.jpg'

    Args:
        clustered_loce_storages (List[List[LoCEMultilayerStorage]]): list of lists of LoCEMultilayerStorage, where external list is clusters, internal lists are leaf LoCEMultilayerStorage
        selected_tag_ids_and_colors (Dict[int, str]): categories (tag_ids) to perform clustering for and corresponding colors, e.g., {3: 'red', 4: 'blue'}
        selected_tag_ids_and_colors (Dict[int, str]): categories (tag_ids) to perform clustering for and corresponding names, e.g., {3: 'car', 4: 'motorcycle'}
        img_out_path (str): output directory for tiled images

    Kwargs:
        image_tile_size (Tuple[int, int]): size of each tile
        model_categories (str = None): different models may have different id-label mappings, needed for correct visualization of bboxes
    """

    mkdir(img_out_path)

    # layers of storage
    if layers == None:
        return

    # number of images per cluster
    cluster_counts = [len(cl) for cl in clustered_loce_storages]

    # evaluation of tile cells
    tile_cells = [find_closest_square_rootable_number(i) for i in cluster_counts]

    
    for layer in layers:

        # for each cluster:    
        for cluster_number, (loce_cluster, tc) in enumerate(zip(clustered_loce_storages, tile_cells)):

            # finalize number of rows and cols
            cols = rows = int(math.sqrt(tc))
            for r in range(rows, 0, -1):
                if (r * cols) >= len(loce_cluster):
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
                    if temp_idx >= len(loce_cluster):
                        continue

                    temp_loce = loce_cluster[temp_idx]

                    # get current image, segmentation and color
                    temp_category = temp_loce.segmentation_category_id
                    temp_category_name = selected_tag_id_names[temp_category]
                    # fix mapping of model categories and MS COCO original categories
                    model_category = temp_category
                    if model_categories is not None:
                        for mc_idx, mc_name in model_categories.items():
                            if mc_name == temp_category_name:
                                model_category = mc_idx
                    temp_image_path = temp_loce.image_path
                    temp_proj = temp_loce.loce_storage[layer].projection
                    temp_seg = temp_loce.segmentation
                    rgb_mask = get_rgb_binary_mask(temp_proj)
                    # temp_selected_predictions = temp_loce.image_predictions
                    # temp_selected_predictions = [pred for pred in temp_loce.image_predictions if int(pred[5]) == model_category]
                    temp_color = selected_tag_ids_and_colors[temp_category]
                    # blend image end segmentation (add colored mask)
                    img_with_seg = blend_imgs(Image.fromarray(get_rgb_binary_mask(temp_proj)), Image.open(temp_image_path), alpha=0.5)
                    img_with_seg_bbox = img_with_seg
                    # img_with_seg_bbox = add_bboxes(img_with_seg, temp_selected_predictions, bbox_color=selected_tag_ids_and_colors[temp_category])
                    # resize before inserting
                    img_with_seg_resized = img_with_seg_bbox.resize(
                        image_tile_size)

                    frame_width = 5

                    # add colored frame
                    draw_frame(ImageDraw.Draw(img_with_seg_resized), image_tile_size, temp_color, frame_width)

                    canvas.paste(img_with_seg_resized, (x_offset, y_offset))

            canvas.save(f'{img_out_path}/cluster_{cluster_number}_{layer}.jpg')


def save_storages_as_tiles_together(loce_storages: Iterable[Any],
                                    img_out_path: str,
                                    model_tag: str,
                                    category_id: int,
                                    image_tile_size: Tuple[int, int] = (180, 135),
                                    frame_width: int = 5,
                                    layers: Iterable[str] = None
                                    ) -> None:

    mkdir(f"{img_out_path}/{model_tag}/")

    # layers of storage
    if layers is None:
        layers = list(loce_storages[0].loce_storage.keys())

    # evaluation of tile cells
    tile_cells = find_closest_square_rootable_number(len(loce_storages))

    for layer in layers:

        save_path = f'{img_out_path}/{model_tag}/storages_{category_id}_{layer}.jpg'
        if os.path.exists(save_path):
            return

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

        # paste into canvas
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

                temp_loce = loce_storages[temp_idx]

                # get current image, segmentation
                temp_image_path = temp_loce.image_path
                temp_proj = temp_loce.loce_storage[layer].projection
                temp_loss = temp_loce.loce_storage[layer].loss
                temp_seg = temp_loce.segmentation

                rgb_mask = get_rgb_binary_mask(temp_proj)

                img_with_proj = blend_imgs(Image.fromarray(rgb_mask), Image.open(temp_image_path), alpha=0.5)
                #img_with_proj_counturs = Image.fromarray(add_countours_around_mask(np.array(img_with_proj), temp_proj, countour_color=(255, 0, 0), thickness=1))
                img_with_proj_resized = img_with_proj.resize(image_tile_size)

                img_with_seg = blend_imgs(get_colored_mask(temp_seg, mask_value_multiplier=255), Image.open(temp_image_path), alpha=0.66)
                img_with_seg_counturs = Image.fromarray(add_countours_around_mask(np.array(img_with_seg), temp_seg, countour_color=(0, 255, 0), thickness=5))
                img_with_seg_resized = img_with_seg_counturs.resize(image_tile_size)

                if temp_loss == 0:
                    temp_color = "black"
                else:
                    temp_color = "white"

                # add colored frame
                draw_frame(ImageDraw.Draw(img_with_proj_resized), image_tile_size, temp_color, frame_width)
                canvas.paste(img_with_proj_resized, (x_offset, y_offset))

                # add colored frame
                draw_frame(ImageDraw.Draw(img_with_seg_resized), image_tile_size, temp_color, frame_width)
                canvas.paste(img_with_seg_resized, (x_offset, y_offset2))

        canvas.save(save_path)


def draw_frame(draw: ImageDraw, image_tile_size: Tuple[int, int], color: str, frame_width: int):
    # top
    draw.line([(0, 0), (image_tile_size[0], 0)],fill=color, width=frame_width)
    # bot
    draw.line([(0, image_tile_size[1] - 1), (image_tile_size[0], image_tile_size[1] - 1)], fill=color, width=frame_width)
    # left
    draw.line([(0, 0), (0, image_tile_size[1])], fill=color, width=frame_width)
    # right
    draw.line([(image_tile_size[0] - 1, 0), (image_tile_size[0] - 1, image_tile_size[1])], fill=color, width=frame_width)


def shorten_layer_name(layer_name: str):
    return layer_name.replace("features", "f").replace("conv_encoder", "ce").replace("encoder_layer_", "el").replace("encoder", "e").replace("layers", "ls").replace("layer", "l").replace("model", "m").replace("backbone", "b").replace("conv", "c").replace("conv_proj", "cp").replace("input_projection", "ip")
