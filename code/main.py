#Import packages
import numpy as np
import pandas as pd

from time import sleep
from pathlib import Path

from fastdownload import download_url
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.optimizers import Adam, SGD
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

import matplotlib.pyplot as plt
import random
from PIL import Image

import shutil
import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import seaborn as sns

import functions

import yaml
import os

import time
import mlflow

import create_yaml_files as yaml_files

#Define classes
class Preprocess:
    def __init__(self, **mdict):
        self.mdict = mdict

    def create_generators(self):
        train_datagen = ImageDataGenerator(
            rescale = self.mdict['generators']['rescale'],
            rotation_range = self.mdict['generators']['rotation_range'],
            width_shift_range = self.mdict['generators']['width_shift_range'],
            height_shift_range = self.mdict['generators']['height_shift_range'],
            shear_range = self.mdict['generators']['shear_range'],
            zoom_range = self.mdict['generators']['zoom_range'],
            fill_mode = self.mdict['generators']['fill_mode'])

        train_generator = train_datagen.flow_from_directory(
            self.mdict['info']['train_dir'],
            target_size = (224, 224),
            batch_size = 32,
            classes = self.mdict['info']['classes'])

        validation_generator = ImageDataGenerator().flow_from_directory(
            self.mdict['info']['validation_dir'],
            target_size = (224, 224),
            batch_size = 32,
            classes = self.mdict['info']['classes'])

        test_generator = ImageDataGenerator().flow_from_directory(
            self.mdict['info']['test_dir'],
            target_size = (224, 224),
            batch_size = 32,
            classes = self.mdict['info']['classes'],
            shuffle = False)

        return train_generator, validation_generator, test_generator
    
class CNN_model:
    def __init__(self, train_generator, validation_generator, **mdict):
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.mdict = mdict

    def define_architecture(self):
        model = Sequential()

        #Convolutional layers
        for i, conv in enumerate(self.mdict['conv_layers']):
            if i==0:
                model.add(Conv2D(
                    filters = conv['filters'],
                    kernel_size = conv['kernel_size'],
                    activation =conv['activation'],
                    input_shape= conv['input_shape']
                ))
            else:
                model.add(Conv2D(
                    filters = conv['filters'],
                    kernel_size = conv['kernel_size'],
                    activation =conv['activation']
                ))

            model.add(MaxPooling2D(self.mdict['maxpool_layers']['pool_size']))

        #Flatten
        model.add(Flatten())

        #Dropout
        model.add(Dropout(self.mdict['model']['dropout']))

        #Dense layers
        model.add(Dense(
            units = self.mdict['dense_layers']['units'],
            activation = self.mdict['dense_layers']['activation']))

        #Output
        model.add(Dense(
            units = self.mdict['output_layer']['units'],
            activation = self.mdict['output_layer']['activation']))

        return model

    def compile_model(self):
        model = self.define_architecture()
        model.compile(
            loss= self.mdict['model']['loss'],
            optimizer= self.mdict['model']['optimizer'],
            metrics=self.mdict['model']['metrics'])
        return model

    def fit_model (self):
        model = self.compile_model()

        log_dir = "../output/logs/" + self.mdict['info']['model_name']
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        history = model.fit(self.train_generator,
                            steps_per_epoch = self.mdict['model']['steps_per_epoch'],
                            epochs = self.mdict['model']['epochs'],
                            validation_data = self.validation_generator, 
                            #validation_steps = self.mdict['model']['validation_steps'],
                            callbacks=[tensorboard_callback]
                            )

        return history, model

    def save_model(self):
        history, model = self.fit_model()
        model.save(self.mdict['info']['model_filepath'])

        with open(self.mdict['info']['history_filepath'], 'wb') as file:
            pickle.dump(history.history, file)

        return history
    
def main()
    #Create and import YAML files
    yaml_files.create_yaml_files()

    #Train models
    start_time = time.time()  # Start timing

    yaml_files = [
        '../input/base_dict.yaml'
    ]

    # Loop through each YAML file
    for yaml_file in yaml_files:
        with open(yaml_file, 'r') as file:
            df_dict = yaml.safe_load(file)

        print(f"Training model with configuration from {yaml_file}: {df_dict}")

        # Create and train the model
        generator = Preprocess(**df_dict)
        train_generator, validation_generator, test_generator = generator.create_generators()
        model = CNN_model(train_generator, validation_generator, **df_dict)
        history = model.save_model()
        print(f"Finished training model with {df_dict}")

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time

if __name__ == "__main__":
    main()