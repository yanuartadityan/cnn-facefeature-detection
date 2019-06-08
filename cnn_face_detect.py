import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras as keras


class Helper:
    @staticmethod
    def is_file(path, ext=None):
        exists = os.path.isfile(path)
        extension = os.path.splitext(path)[-1].lower()

        if exists and (extension == ext):
            return True
        else:
            return False

    @staticmethod
    def show_image(images=[], im_height=96, im_width=96, img_id=0, im_to_show=9, figsize=(8, 8)):
        col = math.ceil(math.sqrt(im_to_show))
        row = col-1 if (col**2 - im_to_show) > col else col 

        fig = plt.figure(figsize=figsize, frameon=False)
        for i in range(1, col*row+1):

            fig.add_subplot(row, col, i)

            if i <= im_to_show:
                img = images[img_id+(i-1), :]

            plt.imshow(img.reshape((im_height, im_width)), cmap='gray')
            plt.axis('off')

        plt.show()

    @staticmethod
    def show_image_wlabels(images=[], labels=[], im_height=96, im_width=96, img_id=0, im_to_show=9, figsize=(8, 8)):

        col = math.ceil(math.sqrt(im_to_show))
        row = col-1 if (col**2 - im_to_show) > col else col 

        fig = plt.figure(figsize=figsize, frameon=False)
        for i in range(1, col*row+1):

            fig.add_subplot(row, col, i)

            if i <= im_to_show:
                img = images[img_id+(i-1), :]
                for j in range(0, labels.shape[1]):
                    if not j&1==1:
                        x_id = labels[img_id+(i-1), j]
                        y_id = labels[img_id+(i-1), j+1]
                        if np.isnan(x_id) or np.isnan(y_id):
                            continue
                        else:
                            img[int(y_id), int(x_id), 0] = 255
            else:
                img = np.zeros((im_height, im_width, 1), dtype=np.float)

            #! TODO: stop casting
            plt.imshow(img.reshape((im_height, im_width)), cmap='gray')
            plt.axis('off')

        plt.show()


class CNNFaceDetect:
    def __init__(self):
        self.Xtrain = []
        self.Ytrain = []
        self.Xtest = []
        self.Ytest = []
        self.supported_dataset = ['montreal']
           
    def load_training_data(self, path=None, dataset='montreal', nrows=None):
        if self.check_dataset(path, dataset):
            if dataset == 'montreal':
                self.im_height = 96
                self.im_width = 96
                self.im_size = (self.im_height, self.im_width, 1)

                if not nrows == None:
                    dframe = pd.read_csv(path, nrows=nrows)
                else:
                    dframe = pd.read_csv(path)

                # fillna
                dframe.fillna(method='ffill', inplace=True)

                # train input
                flatten_image = dframe['Image'].apply(lambda x: np.reshape(x.split(' '), self.im_size).astype(np.float)).values
                self.Xtrain = np.zeros((len(flatten_image), self.im_height, self.im_width, 1), dtype=np.float)

                for i, img in enumerate(flatten_image):
                    self.Xtrain[i, :, :, :] = img

                # train output
                self.Ytrain = dframe.loc[:, dframe.columns != 'Image'].fillna(value=0)
                self.Ytrain = self.Ytrain.values

            if dataset == 'helen':
                raise NotImplementedError

            if dataset == '300w':
                raise NotImplementedError
        else:
            raise FileNotFoundError('provided dataset is not valid or does not exist')

    def load_test_data(self, path=None, dataset='montreal'):
        # handle csv
        if not Helper.is_file(path, ext='.csv'):
            raise ValueError('please provide correct test dataset')
        
        if dataset == 'montreal':
            self.im_height = 96
            self.im_width = 96
            self.im_size = (self.im_height, self.im_width, 1)

            # load test data
            dframe = pd.read_csv(path)
            flatten_image = dframe['Image'].apply(lambda x: np.reshape(x.split(' '), self.im_size).astype(np.float)).values
            self.Xtest = np.zeros((len(flatten_image), self.im_height, self.im_width, 1), dtype=np.float)
            for i, img in enumerate(flatten_image):
                self.Xtest[i, :, :, :] = img
            
        if dataset == 'helen':
            raise NotImplementedError

        if dataset == '300w':
            raise NotImplementedError

    def check_dataset(self, path=None, dataset='montreal'):
        # handle csv
        if not Helper.is_file(path, ext='.csv'):     
            raise ValueError('please provide correct training dataset')
        
        # handle type of dataset
        if not dataset.lower() in self.supported_dataset:
            raise ValueError('check that training dataset is supported')

        return True

    def build_model(self, savepath=None):
        self.model = []

        # keras
        self.model = keras.Sequential([keras.layers.Flatten(input_shape=(self.im_height, self.im_width, 1)),
                                       keras.layers.Dense(128, activation='relu'),
                                       keras.layers.Dropout(0.1),
                                       keras.layers.Dense(64, activation='relu'),
                                       keras.layers.Dense(30)])

        # compile
        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss='mse',
                           metrics=['mae', 'accuracy'])

        # train
        self.model.fit(self.Xtrain, self.Ytrain, epochs=500, batch_size=128, validation_split=0.2)

        # save
        tf.keras.models.save_model(self.model, filepath=savepath, overwrite=True, include_optimizer=True)

    def load_model(self, modelpath=None):
        self.model = []

        # load
        self.model = tf.keras.models.load_model(modelpath, compile=True)

    def predict(self):
        self.Ytest = self.model.predict(self.Xtest)