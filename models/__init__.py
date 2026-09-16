import importlib
from typing import Dict, Type
import torch.nn as nn

from models.base_model import BaseDetector


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str):

    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str, config):

    if name not in MODEL_REGISTRY:

        try:
            module = importlib.import_module(f"models.{name}")

            if hasattr(module, name):
                MODEL_REGISTRY[name] = getattr(module, name)
            else:

                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, nn.Module) and attr is not nn.Module:
                        MODEL_REGISTRY[name] = attr
                        break
        except ImportError as e:
            raise ValueError(f"Unknown model: {name}. Available models: {list(MODEL_REGISTRY.keys())}. Error: {e}")

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available models: {list(MODEL_REGISTRY.keys())}")

    model_cls = MODEL_REGISTRY[name]
    print(f"Creating model: {name} ({model_cls.__name__})")
    return model_cls(config=config)


def list_models():

    return list(MODEL_REGISTRY.keys())


from models.osd import OSD
MODEL_REGISTRY["OSD"] = OSD
