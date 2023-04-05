from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from ares.defender.models import load_pytorch_model, load_tensorflow_model

if TYPE_CHECKING:
    from art.estimators.classification import PyTorchClassifier, TensorFlowV2Classifier


class Classifier:
    def __init__(
        self, framework: str, file: str, name: str, params: dict, dataset_params: dict, checkpoint: Optional[str] = None
    ):
        self.framework = framework
        self.name = name

        input_shape = dataset_params["input_shape"]
        num_classes = dataset_params["num_classes"]
        clip_values = dataset_params.get("clip_values", (0, 1))

        if framework == "pytorch":
            import torch
            from art.estimators.classification import PyTorchClassifier

            self._model = load_pytorch_model(file, name, params, checkpoint)
            self._classifier = PyTorchClassifier(
                model=self._model,
                loss=torch.nn.CrossEntropyLoss(),
                input_shape=input_shape,
                nb_classes=num_classes,
                clip_values=clip_values,
            )
        elif framework == "tensorflow":
            import tensorflow as tf
            from art.estimators.classification import TensorFlowV2Classifier

            self._model = load_tensorflow_model(file, name, params, checkpoint)
            self._classifier = TensorFlowV2Classifier(
                model=self._model,
                loss_object=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                input_shape=input_shape,
                nb_classes=num_classes,
                clip_values=clip_values,
            )
        else:
            raise ValueError(f"Error loading classifier: {framework} framework not supported")

    def reset(self):
        self._model.reset()

    @property
    def classifier(self) -> Union["PyTorchClassifier", "TensorFlowV2Classifier"]:
        return self._classifier

    @property
    def queries(self) -> int:
        return self._model.queries

    def predict(self, x: np.ndarray, logits: bool = False) -> np.ndarray:
        out = self._classifier.predict(x)
        if logits:
            return out
        else:
            y_pred = np.argmax(out, axis=1)
            return y_pred
