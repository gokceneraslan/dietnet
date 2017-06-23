import bcolz
import numpy
from .data_utils import Sequence


class BcolzSequence(Sequence):

    def __init__(self, x_file, y_file, batch_size):
        self.X = bcolz.open(x_file)
        self.y = bcolz.open(y_file)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.X) // self.batch_size

    def __getitem__(self, idx):

        batch_x = self.X[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y = self.y[idx*self.batch_size:(idx+1)*self.batch_size]

        return batch_x, batch_y


