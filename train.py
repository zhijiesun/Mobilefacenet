from archs import *
import keras
from loss import *
import os, shutil
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from load_data import *
#load data
import tensorflow as tf
from keras.callbacks import TensorBoard

base_dir = './data'

train_datagen = ImageDataGenerator(rescale=1./255) #rescale the tensor values to [0,1]
train_image = train_datagen.flow_from_directory(
         base_dir,
         target_size=(112, 96),# I use the photo size (112*96)
         batch_size=256,
         class_mode='categorical')


model_test=mobilefacenet_arcface()
model_test.compile(loss='categorical_crossentropy',
            optimizer=optimizers.SGD(lr=0.01),
            metrics=['acc'])


epochs=10
for e in range(epochs):
    print('Epoch', e)
    for batches in train_image:
        X_batch, label_batch = batches[:]
        print(len(X_batch))
        model_test.fit([X_batch, label_batch],label_batch, epochs=3, batch_size=64)


