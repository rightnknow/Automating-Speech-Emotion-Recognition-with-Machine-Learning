#############Set up###################
# In order to run the script, you have to put all the files under the same directory , and you need to change the path inside the script to your own path where the script locates
# You should also install all the librarys below, I'll try to make up a list in the future
# For the VGG files that I used for feature extraction in LSTM network, They only ask to cite them for public research, but I'm still gonna cite it here
# [1]Ai.google, 2019. [Online]. Available: https://ai.google/research/pubs/pub45857. [Accessed: 13- Mar- 2019].
#

from __future__ import print_function

import numpy as np
from scipy.io import wavfile
import six
import tensorflow as tf
import keras
from keras.models import Sequential,Model
from keras.layers import LSTM, Dense,Dropout,Bidirectional,Input,Concatenate,GlobalAveragePooling2D
from keras.utils import plot_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from vggish_input import wavfile_to_examples
import vggish_postprocess
import vggish_slim
import vggish_params

import os
import glob
from pathlib import Path
import re

import os
import re
import pandas as pd
import pickle
import random
random.seed(1)

temp_dict = {}
embedding_dict = {}
labels = []
image_dir = Path(r'C:\Users\zhanglichuan\Desktop\ECE496\dataimage')
image_list = []
script_path = r'C:\Users\zhanglichuan\Desktop\ECE496\lstm'

image_size = 150


def plot(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()

def train_data():
    with open(os.path.join(script_path,"labels.txt"), "rb") as fp:  # Unpickling
        labels_array = pickle.load(fp)
    with open(os.path.join(script_path,"train_set.txt"), "rb") as fp:  # Unpickling
        audio_set = pickle.load(fp)
    with open(os.path.join(script_path,"image.txt"), "rb") as fp:  # Unpickling
        image_set = pickle.load(fp)



    number_of_feature = audio_set.shape[2]
    timesteps = audio_set.shape[1]
    number_of_class = 6

    train_set = [(audio_set[i],image_set[i]) for i in range(audio_set.shape[0])]

    label_set = keras.utils.to_categorical( labels_array ,num_classes=number_of_class)
    X_train_audio, X_test_audio, X_train_image,X_test_image,Y_train, Y_test = train_test_split(audio_set,image_set, label_set, test_size=0.2, random_state=0)

    X_train_image = np.array(X_train_image)
    X_test_image = np.array(X_test_image)

    X_train_audio = np.array(X_train_audio)
    X_test_audio = np.array(X_test_audio)

    print("load complete")



    #LSTM part
    audioInput = Input(shape=(timesteps,number_of_feature))
    lstmOne = LSTM(32,return_sequences=True)(audioInput)
    dropOne = Dropout(0.3)(lstmOne)
    lstmTwo = LSTM(32)(dropOne)
    dropTwo = Dropout(0.3)(lstmTwo)
    dropTwo = Dense(20)(dropTwo)
    lstmoutputLayer = Dense(6,activation='softmax')(dropTwo)
    lstmModel = Model(input= audioInput ,output = lstmoutputLayer)
    lstmModel.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    history = lstmModel.fit(X_train_audio, Y_train, epochs=20)
    score, accuracy =lstmModel.evaluate(X_test_audio,Y_test)
    print("test accuracy is {}".format(accuracy))
    print("LSTM part done")


    #CNN part
    img_input = Input(shape=(image_size, image_size, 1))
    img_conc = Concatenate()([img_input, img_input, img_input])

    base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=img_conc)
    firstLayer = base_model.output
    firstLayer = GlobalAveragePooling2D()(firstLayer)
    firstLayer = Dense(1000,activation='relu')(firstLayer)
    firstLayer = Dropout(0.5)(firstLayer)
    cnnoutputLayer = Dense(6, activation='softmax')(firstLayer)

    for layer in base_model.layers:
        layer.trainable = False

    cnnModel = Model(input=base_model.input,output = cnnoutputLayer )

    cnnModel.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    #plot_model(model, to_file=r'C:\Users\zhanglichuan\Desktop\ECE496\lstm\model.png', show_shapes=True)
    history = cnnModel.fit(X_train_image, Y_train, epochs=10, validation_split=0.25)

    for layer in cnnModel.layers[:249]:
        layer.trainable = False
    for layer in cnnModel.layers[249:]:
        layer.trainable = True
    from keras.optimizers import SGD
    cnnModel.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
    history = cnnModel.fit(X_train_image, Y_train, epochs=10, validation_split=0.25)


    #plot(history)
    print("On test set")

    score, accuracy =cnnModel.evaluate(X_test_image,Y_test)

    print("test accuracy is {}".format(accuracy))
    #plot_model(model,to_file=r'C:\Users\zhanglichuan\Desktop\ECE496\lstm\model.png',show_shapes=True)
    #print(model.summary())

    print("Start on combining")
    #remove last layer of cnn and lstm network

    CombineLayerOne = cnnModel.layers[-2].output
    CombineLayerTwo = lstmModel.layers[-2].output
    x = keras.layers.concatenate([CombineLayerOne, CombineLayerTwo])
    x = Dropout(0.5)(x)
    x = Dense(6, activation='softmax')(x)
    model = Model(inputs=[img_input, audioInput], outputs=x)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    history = model.fit([X_train_image,X_train_audio],Y_train ,
          epochs=10,validation_split=0.2)
    plot(history)
    #plot_model(model, to_file=r'C:\Users\zhanglichuan\Desktop\ECE496\lstm\Combinemodel.png', show_shapes=True)
    print("On test set")

    score, accuracy =model.evaluate([X_test_image,X_test_audio],Y_test)

    print("test accuracy is {}".format(accuracy))

def main():
    if not (os.path.exists(os.path.join(script_path, 'labels.txt')) and os.path.exists(os.path.join(script_path, 'train_set.txt')) and os.path.exists(os.path.join(script_path, 'image.txt'))):
        preprocess_data()
        #train_data()
    else:
        train_data()

if __name__ == "__main__":
    main()
