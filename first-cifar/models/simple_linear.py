import chainer
import chainer.functions as F
import chainer.links as L

class SimpleLinear(chainer.Chain):
    def __init__(self, class_labels=10):
        super(SimpleLinear, self).__init__()
        with self.init_scope():
            self.fc1 = L.Linear(None, class_labels)
        
    def __call__(self, x):
        return self.fc1(x)
