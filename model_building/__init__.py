import os
import kfp
from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics, ClassificationMetrics

@component(base_image="tensorflow/tensorflow")
def model_building(ml_model : Output[Model]):
    '''
    Define the model architecture
    This way it's more simple to change the model architecture and all the steps and indipendent
    '''
    from tensorflow import keras
    import tensorflow as tf
    import os
    
    #model definition
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28,28,1)))
    model.add(keras.layers.MaxPool2D(2, 2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))

    model.add(keras.layers.Dense(10, activation='softmax'))
    
    #saving model
    model.save(ml_model.path)