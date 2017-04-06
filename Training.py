import Generator
from keras.layers import Input, merge
from keras.layers import Dense, Dropout, Convolution2D, Flatten, Reshape, Activation
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.models import Model, Sequential
from keras.models import load_model
from keras.optimizers import SGD, Adagrad, Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.callbacks import CSVLogger
import numpy as np
import pandas as pd
import random
from keras.layers.core import SpatialDropout2D
from sys import argv
from lidar_bv_loss import multitask_loss


def regularizer(wreg):
    if wreg > 0.0:
        return l2(wreg)
    else:
        return None


class Training(object):
    def __init__(self, batch_size=32, num_classes=1, num_parameters=7, input_shape=[512,512,3]):
        self.model = Sequential()
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.current_shape = input_shape
        self.is_first_layer = True
        self.is_first_dense_layer = True
        self.winit = 'glorot_normal'
        self.wreg = 0.01
        self.use_batchnorm = True
        self.output_file_stem = "models/" + argv[0].split(".")[0].replace("train-","")
        self.generator_options = {}
        self.lr = 0.01
        self.model_checkpoint = ModelCheckpoint(self.output_file_stem + ".hdf5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
        self.csv_logger = CSVLogger(self.output_file_stem + ".log")


    def callbacks(self, options):
        result = []
        patience = options.get('lr_patience', 4)
        factor = options.get('lr_factor', 0.5)
        cooldown = options.get('lr_cooldown', 4)
        min_lr = options.get('lr_min', 0)
        reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience, verbose=1, mode='auto', epsilon=0.0001, cooldown=cooldown, min_lr=min_lr)
        result.append(reduce_learning_rate)

        for c in (self.model_checkpoint, self.csv_logger):
            if not c is None:
                result.append(c)
        return result


    def compile(self, loss_function):
        self.model.compile(
            optimizer=Adam(self.lr),
            loss=loss_function)


    def conv(self, depth, filter_size=3, subsample=(1,1)):
        conv_layer = Convolution2D(depth, filter_size, filter_size, border_mode='same', W_regularizer=regularizer(self.wreg), input_shape=self.current_shape, subsample=subsample, init=self.winit)
        self.is_first_layer = False
        self.model.add(conv_layer)
        if self.use_batchnorm:
            self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.current_shape[2] = depth


    def dense(self, output_size):
        if self.is_first_layer:
            input_shape = self.input_shape
            self.is_first_layer = False
            self.is_first_dense_layer = False

        if self.is_first_dense_layer:
            self.model.add(Flatten())
            self.is_first_dense_layer = False

        self.model.add(Dense(output_size, init='normal', W_regularizer=regularizer(self.wreg)))

        if self.use_batchnorm:
            self.model.add(BatchNormalization())

        self.model.add(Activation('relu'))


    def classifier(self, nx=32, ny=32, num_parameters=7):
        self.model.add(Dense(nx*ny*num_parameters, init=self.winit))


    def binary_classifier(self):
        self.model.add(Dense(1, init=self.winit))
        self.model.add(Activation('sigmoid'))


    def sigmoid(self):
        self.model.add(Activation('sigmoid'))


    def maxpool(self):
        self.model.add(MaxPooling2D())
        self.current_shape[0] /= 2
        self.current_shape[1] /= 2


    def avgpool(self):
        self.model.add(AveragePooling2D())
        self.current_shape[0] /= 2
        self.current_shape[1] /= 2


    def dropout(self, p):
        self.model.add(Dropout(p))


    def train(self, options={}):
        epoch_offset = 0

        self.compile(multitask_loss)
        num_epochs = options.get('num_epochs', 1000)
        num_training = options.get('num_training', None)
        num_validation = options.get('num_validation', 2048)

        file_stems = Generator.get_file_stems("training")
        file_stems_train, file_stems_val = Generator.split_test_set(file_stems)
        generator_train = Generator.Generator(file_stems_train, batch_size=self.batch_size)
        generator_val = Generator.Generator(file_stems_val, batch_size=self.batch_size, augment_data=False)

        self.model.fit_generator(
            generator_train, len(file_stems_train), num_epochs,
            validation_data = generator_val,
            nb_val_samples = len(file_stems_val),
            callbacks = self.callbacks(options),
            max_q_size=8, nb_worker=8, pickle_safe=True)  # starts training
