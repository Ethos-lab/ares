from typing import Optional, Tuple

import numpy as np
from art.config import set_data_path
from art.utils import load_dataset


class Dataset:
    def __init__(self, name: str, dataroot: Optional[str] = None, train_set: bool = False, channels_first: bool = True):
        self.name = name
        self.dataroot = dataroot
        self.train_set = train_set
        self.channels_first = channels_first

        if dataroot is not None:
            set_data_path(dataroot)

        (x_train, y_train), (x_test, y_test), _, _ = load_dataset(name)
        if train_set:
            self.x = x_train
            self.y = y_train
        else:
            self.x = x_test
            self.y = y_test

        if channels_first:
            self.x = np.transpose(self.x, (0, 3, 1, 2))

        self.x = self.x.astype(np.float32)
        self.y = np.argmax(self.y, axis=1)
        self.size = len(self.x)

        self.permutation = np.random.permutation(self.size)
        self.index = 0

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        index = self.permutation[self.index]
        self.index += 1
        if self.index >= self.size:
            self.index = 0

        x = self.x[[index]]
        y = self.y[[index]]
        return x, y
