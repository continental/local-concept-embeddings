'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

from abc import ABC, abstractmethod
from typing import Iterable, Union


class Propagator(ABC):

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def set_layers(self, layers: Union[str, Iterable[str]]) -> None:
        pass

    @abstractmethod
    def get_predictions(self) -> None:
        pass

    @abstractmethod
    def get_activations(self) -> None:
        pass

    @abstractmethod
    def get_gradients(self) -> None:
        pass


class PropagatorTransformer(Propagator):

    pass