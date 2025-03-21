'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

from .hooks import ForwardHook, BackwardHook, ForwardInsertHook
from .propagator import Propagator, PropagatorTransformer
from .propagator_torch_classifier import PropagatorTorchClassifier
from .propagator_torch_detector import PropagatorTorchDetector
from .propagator_ultralytics_yolo_old import PropagatorUltralyticsYOLOv3Old, PropagatorUltralyticsYOLOv5Old
from .propagator_torch_rcnn import PropagatorTorchRCNN
from .propagator_torch_ssd import PropagatorTorchSSD
from .propagator_torch_retinanet import PropagatorTorchRetinaNet
from .propagator_huggingface_detr import PropagatorHuggingFaceDETR

DETECTION_PROPAGATORS = {
    'rcnn': PropagatorTorchRCNN,
    'yolo3': PropagatorUltralyticsYOLOv3Old,
    'yolo5': PropagatorUltralyticsYOLOv5Old,
    'ssd': PropagatorTorchSSD,
    'retinanet': PropagatorTorchRetinaNet,
    'detr': PropagatorHuggingFaceDETR
}
