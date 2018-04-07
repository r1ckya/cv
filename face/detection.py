from matplotlib import pyplot as plt
from glob import glob
import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import tensorflow as tf
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from skimage.util import random_noise, crop
from sklearn.utils import shuffle
from skimage.transform import resize, rotate
from os.path import join, split

IMSIZE = 100
flip_indices = [
                (0, 6), (1, 7), (2, 4), (3, 5),
                (10, 16), (11, 17), (12, 14), (13, 15),
                (8, 18), (9, 19), (22, 26), (23, 27)
                ]

class Generator(object):
    import numpy as np
    def __init__(self,
                 X, 
                 y,
                 bs,
                 flip_ratio,
                 rotate_ratio,
                 noise_ratio,
                 zoom_ratio,
                 zoom_range,
                 flip_indices):
        self.X = X
        self.y = y
        self.bs = bs
        self.flip_ratio = flip_ratio
        self.rotate_ratio = rotate_ratio
        self.noise_ratio = noise_ratio
        self.zoom_ratio = zoom_ratio
        self.zoom_range = zoom_range
        
        self.size = X.shape[0]
        self.flip_indices = flip_indices
    
    def _random_indices(self, ratio):
        size = int(self.bs * ratio)
        return np.random.choice(self.bs, size, replace=False)
    
    def flip(self):
        indices = self._random_indices(self.flip_ratio)
        self.inputs[indices] = self.inputs[indices, :, ::-1]
        self.targets[indices, ::2] = self.targets[indices, ::2] * -1
        for a, b in self.flip_indices:
            self.targets[indices, a], self.targets[indices, b] = \
            self.targets[indices, b], self.targets[indices, a]
    
    def rotate(self):
        indices = self._random_indices(self.rotate_ratio)
        self.targets = self.targets.reshape(len(self.targets), self.y.shape[1] // 2, 2)
        for i in indices:
            angle = np.random.randint(-10, 10)
            self.inputs[i] = rotate(self.inputs[i], angle)
            angle = angle * np.pi / 180
            C = [[np.cos(angle), -np.sin(angle)],
                 [np.sin(angle), np.cos(angle)]]
            self.targets[i] = np.dot(self.targets[i], C)
        self.targets = self.targets.reshape(-1, self.y.shape[1])
    
    def zoom(self):
        indices = self._random_indices(self.zoom_ratio)
        for i in indices:
            a, b = np.random.randint(0, self.zoom_range, 2)
            self.targets[i] = self.targets[i] * (IMSIZE / 2) + (IMSIZE / 2)
            self.targets[i, ::2] = self.targets[i, ::2] - b
            self.targets[i, 1::2] = self.targets[i, 1::2] - a
            self.targets[i] = 2 * (self.targets[i] - (IMSIZE - self.zoom_range) / 2) / (IMSIZE - self.zoom_range)
            self.inputs[i] = resize(self.inputs[i, a:-self.zoom_range+a, b:-self.zoom_range+b], (IMSIZE, IMSIZE))
            
    def noise(self):
        indices = self._random_indices(self.noise_ratio)
        for i in indices:
            self.inputs[i] = random_noise(self.inputs[i])
    
    def generate(self):
        while True:
            self.X, self.y = shuffle(self.X, self.y)
            start = 0
            stop = self.bs
            for i in range(self.size // self.bs):
                self.inputs = self.X[start:stop].copy()
                self.targets = self.y[start:stop].copy()
                start += self.bs
                stop += self.bs
                self.flip()
                self.rotate()
                self.noise()
                self.zoom()
                yield (self.inputs, self.targets)
                
def CNN():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(IMSIZE, IMSIZE, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(28))
    
    return model

def cosine_lr(start, stop, epochs, n):
    epochs /= n
    res = stop + (start - stop) / 2 * (1  + np.cos(np.linspace(0, epochs, epochs) * np.pi / epochs))
    res = np.concatenate([res for i in range(n)])
    return res

def train_detector(y_dict, img_dir, fast_train=True):
    
    model = CNN()
    model.compile(loss='mse', optimizer='adam')
    
    mega_bs = 1000
    items = list(y_dict.items())
    for i in range(int(np.ceil(len(y_dict) / mega_bs))):
        bs = min(len(y_dict) - i * mega_bs, mega_bs)
        y = np.empty((bs, 28))
        X = np.empty((bs, IMSIZE, IMSIZE, 3))
        for j, (k, v) in enumerate(items[i * mega_bs: (i + 1) * mega_bs]):
            y[j] = v
            img = img_to_array(load_img(join(img_dir, k))) / 255
            h, w = img.shape[:2]
            X[j] = resize(img, (IMSIZE, IMSIZE))
            y[j, 0::2] = 2 * (y[j, 0::2] - h / 2) / h
            y[j, 1::2] = 2 * (y[j, 1::2] - w / 2) / w
        model.fit(X, y, batch_size=100, validation_split=0.2)
        
    
def detect(model, img_dir):
    
    fnames = glob(join(img_dir, '*'))
    y_dict = dict()
    mega_bs = 1000
    for i in range(int(np.ceil(len(fnames) / mega_bs))):
        bs = min(len(fnames) - i * mega_bs, mega_bs)
        sz = np.empty((bs, 2))
        X = np.empty((bs, IMSIZE, IMSIZE, 3))
        for j, name in enumerate(fnames[i * mega_bs: (i + 1) * mega_bs]):
            img = img_to_array(load_img(name)) / 255
            sz[j] = img.shape[:2]
            X[j] = resize(img, (IMSIZE, IMSIZE))
            
        y = model.predict(X, batch_size=100)
        
        for j, name in enumerate(fnames[i * mega_bs: (i + 1) * mega_bs]):
            h, w = sz[j]
            y[j, 0::2] = y[j, 0::2] * h / 2 + h / 2
            y[j, 1::2] = y[j, 1::2] * w / 2 + w / 2
            y_dict[split(name)[-1]] = y[j]
            
    return y_dict
