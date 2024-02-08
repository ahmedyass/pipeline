from kfp.dsl import component, Output, Dataset

@component(
        packages_to_install=['tensorflow==2.4.0', 'numpy']
)
def preprocess_data(
    training_data_output: Output[Dataset],
    test_data_output: Output[Dataset]
):
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist
    import numpy as np
    import os

    # Load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Save preprocessed data to the component's output paths
    np.savez_compressed(training_data_output.path, x_train=x_train, y_train=y_train)
    np.savez_compressed(test_data_output.path, x_test=x_test, y_test=y_test)
