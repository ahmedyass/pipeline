from kfp.dsl import component, Input, Output, Dataset, Metrics

@component(
        packages_to_install['numpy']
)
def preprocessing(metrics : Output[Metrics], x_train_processed : Output[Dataset], x_test_processed: Output[Dataset],
                  x_train_artifact: Input[Dataset], x_test_artifact: Input[Dataset]):
    ''' 
    just reshape and normalize data
    '''
    import numpy as np
    import os
    
    # load data artifact store
    x_train = np.load(x_train_artifact.path) 
    x_test = np.load(x_test_artifact.path)
    
    # reshaping the data
    # reshaping pixels in a 28x28px image with greyscale, canal = 1. This is needed for the Keras API
    x_train = x_train.reshape(-1,28,28,1)
    x_test = x_test.reshape(-1,28,28,1)
    # normalizing the data
    # each pixel has a value between 0-255. Here we divide by 255, to get values from 0-1
    x_train = x_train / 255
    x_test = x_test / 255
    
    #logging metrics using Kubeflow Artifacts
    metrics.log_metric("Len x_train", x_train.shape[0])
    metrics.log_metric("Len y_train", x_test.shape[0])
   
    
    # save feuture in artifact store
    np.save("tmp/x_train.npy",x_train)
    os.rename("tmp/x_train.npy", x_train_processed.path)
    
    np.save("tmp/x_test.npy",x_test)
    os.rename("tmp/x_test.npy", x_test_processed.path)