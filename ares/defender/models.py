import importlib
import importlib.util
import os
from typing import Optional


def load_pytorch_model(file: str, name: str, params: dict, checkpoint: Optional[str] = None):
    import torch

    class PyTorchModelWrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module):
            super().__init__()
            self.model = model
            self.queries = 0

        def reset(self):
            self.queries = 0

        def forward(self, x: torch.Tensor):
            self.queries += 1
            return self.model(x)

    # load pytorch model from file
    spec = importlib.util.spec_from_file_location("module.name", file)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ctor = getattr(module, name)
        model: torch.nn.Module = ctor(**params)
    else:
        raise ImportError(f"Error loading classifier: cannot load {name} from {file}")

    # load model weights if provided
    if checkpoint is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)

    wrapper = PyTorchModelWrapper(model)
    return wrapper


def load_tensorflow_model(file: str, name: str, params: dict, checkpoint: Optional[str] = None):
    import tensorflow as tf

    class TensorFlowModelWrapper(tf.keras.Model):
        def __init__(self, model: tf.keras.Model):
            super().__init__()
            self.model = model
            self.queries = 0

        def reset(self):
            self.queries = 0

        def call(self, x: tf.Tensor):
            self.queries += 1
            return self.model(x)

    # if checkpoint is a directory, load the entire model
    if checkpoint and os.path.isdir(checkpoint):
        full_model = tf.keras.models.load_model(checkpoint)
        wrapper = TensorFlowModelWrapper(full_model)
        return wrapper

    # load tensorflow model from file
    spec = importlib.util.spec_from_file_location("module.name", file)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ctor = getattr(module, name)
        model: tf.keras.Model = ctor(**params)
    else:
        raise ImportError(f"Error loading classifier: cannot load {name} from {file}")

    # load model weights if provided
    if checkpoint is not None:
        model.load_weights(checkpoint)

    wrapper = TensorFlowModelWrapper(model)
    return wrapper
