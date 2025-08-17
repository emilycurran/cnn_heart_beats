import tensorflow as tf  
from tensorflow.keras import layers, models
from tensorflow.keras.initializers import GlorotUniform

def build_model(input_shape):
    model = models.Sequential()

    # layer1: conv
    model.add(layers.Conv2D(64, (3, 3), strides=1, input_shape=input_shape, padding='same', kernel_initializer=GlorotUniform()))
    model.add(layers.Activation('elu'))  
    model.add(layers.BatchNormalization())  

    # layer2: conv
    model.add(layers.Conv2D(64, (3, 3), strides=1, activation=None, padding='same', kernel_initializer=GlorotUniform()))
    model.add(layers.Activation('elu'))  
    model.add(layers.BatchNormalization())  

    model.add(layers.MaxPooling2D((2, 2), strides=2))

    # layer3: conv
    model.add(layers.Conv2D(128, (3, 3), strides=1, activation=None, padding='same', kernel_initializer=GlorotUniform()))
    model.add(layers.Activation('elu'))  
    model.add(layers.BatchNormalization())  

    # layer4: conv
    model.add(layers.Conv2D(128, (3, 3), strides=1, activation=None, padding='same', kernel_initializer=GlorotUniform()))
    model.add(layers.Activation('elu'))  
    model.add(layers.BatchNormalization())  

    model.add(layers.MaxPooling2D((2, 2), strides=2))

    # layer5: conv
    model.add(layers.Conv2D(256, (3, 3), strides=1, activation=None, padding='same', kernel_initializer=GlorotUniform()))
    model.add(layers.Activation('elu'))  
    model.add(layers.BatchNormalization())  

    # layer6: conv
    model.add(layers.Conv2D(256, (3, 3), strides=1, activation=None, padding='same', kernel_initializer=GlorotUniform()))
    model.add(layers.Activation('elu'))  
    model.add(layers.BatchNormalization())  

    model.add(layers.MaxPooling2D((2, 2), strides=2))

    # layer7: fully connected dense layer
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation=None, kernel_initializer=GlorotUniform()))
    model.add(layers.Activation('elu'))  
    model.add(layers.BatchNormalization())  
    model.add(layers.Dropout(0.5))  

    # layer8: output layer
    model.add(layers.Dense(8, activation='softmax', kernel_initializer=GlorotUniform()))

    model.summary()
    return model
