from kfp.dsl import component, Output, Dataset

@component(
        base_image="tensorflow/tensorflow"
)
def load_dataset(x_train_artifact: Output[Dataset],
                 x_test_artifact: Output[Dataset],
                 y_train_artifact: Output[Dataset],
                 y_test_artifact: Output[Dataset]
    ):
    '''
    get dataset from Keras and load it separating input from output and train from test
    '''
    import numpy as np
    from tensorflow import keras
    import os
   
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    np.save("/tmp/x_train.npy",x_train)
    os.rename("/tmp/x_train.npy", x_train_artifact.path)
    
    np.save("/tmp/y_train.npy",y_train)
    os.rename("/tmp/y_train.npy", y_train_artifact.path)
    
    np.save("/tmp/x_test.npy",x_test)
    os.rename("/tmp/x_test.npy", x_test_artifact.path)
    
    np.save("/tmp/y_test.npy",y_test)
    os.rename("/tmp/y_test.npy", y_test_artifact.path)
