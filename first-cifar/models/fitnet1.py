import chainer
import chainer.functions as F
import chainer.links as L

class FitNet1(chainer.Chain):
    def __init__(self, class_labels=10):
        super(FitNet1, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3,16,ksize=(3,3),pad=1)
            self.conv1_2 = L.Convolution2D(16,16,ksize=(3,3),pad=1)
            self.conv1_3 = L.Convolution2D(16,16,ksize=(3,3),pad=1)

            self.conv2_1 = L.Convolution2D(16,32,ksize=(3,3),pad=1)
            self.conv2_2 = L.Convolution2D(32,32,ksize=(3,3),pad=1)
            self.conv2_3 = L.Convolution2D(32,32,ksize=(3,3),pad=1)

            self.conv3_1 = L.Convolution2D(32,48,ksize=(3,3),pad=1)
            self.conv3_2 = L.Convolution2D(48,48,ksize=(3,3),pad=1)
            self.conv3_3 = L.Convolution2D(48,64,ksize=(3,3),pad=1)
    
            self.fc1 = L.Linear(64, class_labels)

    def __call__(self, x):
        for layer in (self.conv1_1, self.conv1_2, self.conv1_3):
            x = layer(x)
        x = F.max_pooling_2d(x, ksize=2)
        for layer in (self.conv2_1, self.conv2_2, self.conv2_3):
            x = layer(x)
        x = F.max_pooling_2d(x, ksize=2)
        for layer in (self.conv3_1, self.conv3_2, self.conv3_3):
            x = layer(x)
        x = F.average_pooling_2d(x, ksize=8)
        return self.fc1(x)
