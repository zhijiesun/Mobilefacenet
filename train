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
         batch_size=50,
         class_mode='categorical')


model_test=mobilefacenet_arcface()
model_test.compile(loss='categorical_crossentropy',
            optimizer=optimizers.SGD(lr=0.01),
            metrics=['acc'])

batchon=0
for batch in train_image:
    batch=train_image[0]
    X_batch,label_batch=batch[:]
    loss=model_test.train_on_batch([X_batch,label_batch],label_batch)
    print("batchon batch:"+str(loss))


