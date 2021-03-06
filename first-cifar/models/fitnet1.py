import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I

import numpy as np

initW = I.Orthogonal(dtype=np.float32)

class FitNet1(chainer.Chain):
    def __init__(self, class_labels=10):
        super(FitNet1, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3,16,ksize=(3,3),pad=1,initialW=initW)
            self.conv1_2 = L.Convolution2D(16,16,ksize=(3,3),pad=1,initialW=initW)
            self.conv1_3 = L.Convolution2D(16,16,ksize=(3,3),pad=1,initialW=initW)

            self.conv2_1 = L.Convolution2D(16,32,ksize=(3,3),pad=1,initialW=initW)
            self.conv2_2 = L.Convolution2D(32,32,ksize=(3,3),pad=1,initialW=initW)
            self.conv2_3 = L.Convolution2D(32,32,ksize=(3,3),pad=1,initialW=initW)

            self.conv3_1 = L.Convolution2D(32,48,ksize=(3,3),pad=1,initialW=initW)
            self.conv3_2 = L.Convolution2D(48,48,ksize=(3,3),pad=1,initialW=initW)
            self.conv3_3 = L.Convolution2D(48,64,ksize=(3,3),pad=1,initialW=initW)
    
            self.fc1 = L.Linear(64, class_labels)

    def __call__(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.relu(self.conv1_3(x))
        x = F.max_pooling_2d(x, ksize=2)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.relu(self.conv2_3(x))
        x = F.max_pooling_2d(x, ksize=2)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.average_pooling_2d(x, ksize=8)
        return self.fc1(x)
