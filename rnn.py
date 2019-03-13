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



sample1 = wavfile_to_examples(r'C:\Users\zhanglichuan\Desktop\ECE496\lstm\03-01-01-01-01-01-01.wav')
sample2 = wavfile_to_examples(r'C:\Users\zhanglichuan\Desktop\ECE496\lstm\03-01-01-01-01-01-01-seg0.wav')
#sample3 = wavfile_to_examples(r'C:\Users\zhanglichuan\Desktop\ECE496\lstm\03-01-01-01-01-01-24-seg0.wav')
#with tf.Graph().as_default(), tf.Session() as sess:
#    # Define the model in inference mode, load the checkpoint, and
#    # locate input and output tensors.
#    vggish_slim.define_vggish_slim(training=False)
#    vggish_slim.load_vggish_slim_checkpoint(sess, r'C:\Users\zhanglichuan\Desktop\ECE496\lstm\vggish_model.ckpt')
#    features_tensor = sess.graph.get_tensor_by_name(
#        vggish_params.INPUT_TENSOR_NAME)
#    embedding_tensor = sess.graph.get_tensor_by_name(
#        vggish_params.OUTPUT_TENSOR_NAME)
#
#    # Run inference and postprocessing.
#    [embedding_batch] = sess.run([embedding_tensor],
#                                 feed_dict={features_tensor: sample3})
#pproc = vggish_postprocess.Postprocessor(r'C:\Users\zhanglichuan\Desktop\ECE496\lstm\vggish_pca_params.npz')
#sample4 =pproc.postprocess(embedding_batch)
#print("haha")




###global varaible###
temp_dict = {}
embedding_dict = {}
labels = []
image_dir = Path(r'C:\Users\zhanglichuan\Desktop\ECE496\dataimage')
image_list = []
script_path = r'C:\Users\zhanglichuan\Desktop\ECE496\lstm'

image_size = 150

#####################


def get_emotion_label(filename):
    EMOTION_LABEL_POS = 2
    return int(re.findall(r"\d+", os.path.basename(filename))[EMOTION_LABEL_POS]) - 1



def preprocess_data():
    audio_root_dir = Path(r'C:\Users\zhanglichuan\Desktop\ECE496\data')
    audio_file_pattern = Path(r'**/*.wav')
    # takes about 6-8 min on my machine
    counter = 0
    oldm,oldn =0,0



    for audio_file in glob.iglob(str(audio_root_dir / audio_file_pattern), recursive=True):
        #load label
        sample = wavfile_to_examples(audio_file)
        #print(audio_file)
        image_path = re.sub('.wav','.jpg',os.path.split(audio_file)[1])
        image_path = os.path.join(image_dir,image_path)
        #print(image_path)
        input_image = load_img(image_path,target_size=(image_size,image_size),color_mode='grayscale')
        input_image = img_to_array(input_image)
        #input_image = np.expand_dims(input_image, axis=0)
        input_image = preprocess_input(input_image)
        if sample.shape[0] == 0 or get_emotion_label(audio_file) == 0 or get_emotion_label(audio_file) == 1:
            continue
        else:
            labels.append(get_emotion_label(audio_file)-2)
            temp_dict[counter] = sample
            image_list.append(input_image);
        if counter % 100 == 0:
            print('Processing the {}th file: {}'.format(counter, audio_file))
        counter += 1

    oldm,oldn = 0,0
    check = temp_dict
    print("start to construct embedding feature from input")
    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, r'C:\Users\zhanglichuan\Desktop\ECE496\lstm\vggish_model.ckpt')
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)

        # Run inference and postprocessing.
        counter = 0
        for key in temp_dict:
            #print(counter)
            #print(temp_dict[key])
            [embedding_batch] = sess.run([embedding_tensor],
                                         feed_dict={features_tensor: temp_dict[key]})
            embedding_dict[key] = embedding_batch
            m,n = embedding_batch.shape[0],embedding_batch.shape[1]
            if m>oldm:
                oldm = m
            if n > oldn:
                oldn = n
            if counter % 100 == 0:
                print('Processing the {}th file: {}'.format(counter, audio_file))
            counter += 1
    maxLen = oldm*oldn
    pproc = vggish_postprocess.Postprocessor(r'C:\Users\zhanglichuan\Desktop\ECE496\lstm\vggish_pca_params.npz')
    train_set = []
    counter = 0
    for key in embedding_dict:
        print(key)
        test = embedding_dict[key]
        embed_sample = embedding_dict[key].flatten()
        tempOne = np.pad(embed_sample,(0,maxLen-embed_sample.shape[0]), mode='constant', constant_values = 0 )
        temp_embed = np.reshape(tempOne,(1,oldm,oldn))

        if counter == 0:
            train_set = temp_embed
        else:
            train_set = np.concatenate((train_set,temp_embed),axis=0)
        if counter % 100 == 0:
            print('Processing the {}th file: {}'.format(counter, audio_file))
        counter += 1
    print("preprocess finished")

    with open(os.path.join(script_path,'labels.txt'), 'wb') as tfp:
        pickle.dump(labels, tfp)
    with open(os.path.join(script_path,'train_set.txt'),'wb') as tdfp:
        pickle.dump(train_set,tdfp)
    with open(os.path.join(script_path,'image.txt'),'wb') as imfp:
        pickle.dump(image_list, imfp)


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

    """
    model = Sequential()
    model.add(LSTM(32,return_sequences=True,input_shape=(timesteps,number_of_feature)))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dropout(0.3))
    model.add(Dense(6,activation='softmax'))

    """


    #LSTM part
    audioInput = Input(shape=(timesteps,number_of_feature))
    lstmOne = LSTM(32,return_sequences=True)(audioInput)
    dropOne = Dropout(0.3)(lstmOne)
    lstmTwo = LSTM(32)(dropOne)
    dropTwo = Dropout(0.3)(lstmTwo)
    #dropTwo = Dense(20)(dropTwo)
    outputLayer = Dense(6,activation='softmax')(dropTwo)
    lstmModel = Model(input= audioInput ,output = outputLayer)
    lstmModel.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    history = lstmModel.fit(X_train_audio, Y_train, epochs=20, validation_split=0.25)
    score, accuracy =lstmModel.evaluate(X_test_audio,Y_test)
    lstmModel.save('lstm.h5')
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
    outputLayer = Dense(6, activation='softmax')(firstLayer)

    for layer in base_model.layers:
        layer.trainable = False

    cnnModel = Model(input=base_model.input,output = outputLayer )

    cnnModel.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    history = cnnModel.fit(X_train_image, Y_train, epochs=10, validation_split=0.25)

    for layer in cnnModel.layers[:249]:
        layer.trainable = False
    for layer in cnnModel.layers[249:]:
        layer.trainable = True
    from keras.optimizers import SGD
    cnnModel.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
    history = cnnModel.fit(X_train_image, Y_train, epochs=10, validation_split=0.25)
    cnnModel.save('cnn.h5')

    plot(history)
    print("On test set")

    score, accuracy =cnnModel.evaluate(X_test_image,Y_test)

    print("test accuracy is {}".format(accuracy))
    #plot_model(model,to_file=r'C:\Users\zhanglichuan\Desktop\ECE496\lstm\model.png',show_shapes=True)
    #print(model.summary())

    print("Start on combining")



def main():
    if not (os.path.exists(os.path.join(script_path, 'labels.txt')) and os.path.exists(os.path.join(script_path, 'train_set.txt')) and os.path.exists(os.path.join(script_path, 'image.txt'))):
        preprocess_data()
        #train_data()
    else:
        train_data()

if __name__ == "__main__":
    main()