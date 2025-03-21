'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

from typing import Any, Dict, List, Union, Iterable

import torch
from torch import Tensor

from hooks import ForwardHook, PropagatorTransformer
from xai_utils.nn import get_module_layer_by_name


class PropagatorHuggingFaceDETR(PropagatorTransformer):

    def __init__(self,
                 model: Any,
                 layers: Union[str, Iterable[str]],
                 batch_size: int = 16,
                 device: torch.device = torch.device(
                     "cuda" if torch.cuda.is_available() else "cpu")
                 ) -> None:
        """
        Args:
            model: model
            layers: list of layers for activations and/or gradients registration
            batch_size: butch size for sampling from AbstractDataset instances

        Kwargs:
            batch_size: batch size for conversion of AbstractDataset to DataLoader, default value is 32
            device: torch device
        """
        # originally downloaded, wrapped for data input and output
        self.device = device
        self.model = model.to(self.device)
        self.batch_size = batch_size

        self.set_layers(layers)

    def set_layers(self, layers: Union[str, Iterable[str]]) -> None:
        """
        Args:
            layers: list of layers for activations and/or gradients registration
        """
        if isinstance(layers, str):
            layers = [layers]

        self.layers = layers
        self.modules = {layer: get_module_layer_by_name(self.model, layer) for layer in self.layers}

    def get_predictions(self,
                        encoding: Dict
                        ) -> List[Tensor]:
        """
        Propagates the input through the network to get predictions

        Args:
            encoding: encoding batch tensor

        Returns:
            predictions (dims -> 0:3 - bbox, 4 - probability, 5 - class): List[Tensor[N_pred, 6]]
        """
        with torch.no_grad():
            self.model.eval()

            #device_input = input.to(self.device)
            for key, value in encoding.items():
                encoding[key] = value.to(self.device)

            pred = self.model(**encoding, output_attentions=True)
            return pred

    def get_activations(self,
                        encoding: Dict
                        ) -> Dict[str, Tensor]:
        """
        Propagate forward and get activations

        Args:
            encoding: encoding batch tensor

        Returns:
            dictionary - {layer: activations}: Dict[str, Tensor[...]]
        """
        with torch.no_grad():
            self.model.eval()

            fhooks = [ForwardHook(l, m) for l, m in self.modules.items()]

            #device_input = input.to(self.device)
            for key, value in encoding.items():
                encoding[key] = value.to(self.device)

            self.model(**encoding)

            activations = {fh.layer_name: fh.storage[0] for fh in fhooks}

            fhooks = [fh.remove() for fh in fhooks]

            return activations
        
    def get_gradients(self) -> None:
        raise NotImplementedError("Not implemented")