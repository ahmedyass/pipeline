# data_processing.py
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

# model_training.py
from kfp.v2.dsl import component, Input, Output, Model, Dataset

@component(
        packages_to_install=['tensorflow==2.4.0', 'numpy']
)
def train_model(
    training_data: Input[Dataset],
    model_output: Output[Model]
):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.utils import to_categorical
    import numpy as np

    # Load preprocessed training data
    with np.load(training_data.path) as data:
        x_train = data['x_train']
        y_train = data['y_train']
    y_train = to_categorical(y_train)

    # Define and compile the model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=5)

    # Save the trained model
    model.save(model_output.path)

# model_evaluation.py
from kfp.dsl import component, Input, Model

@component(
        packages_to_install=['tensorflow==2.4.0']
)
def evaluate_model(
    model_input: Input[Model]
    ):
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical

    # Load and preprocess test dataset
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test / 255.0
    y_test = to_categorical(y_test)

    # Load the model from the input path provided by Kubeflow
    model = tf.keras.models.load_model(model_input.path)

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# pipeline.py
from kfp.dsl import pipeline
import kfp.compiler as compiler
from data_loading import preprocess_data
from model_evaluation import evaluate_model
from model_training import train_model

@pipeline(
  name='mnist-classification-pipeline',
  description='A pipeline that processes MNIST data and trains a model.'
)
def mnist_pipeline():
    preprocess_task = preprocess_data()
    train_task = train_model(
        training_data=preprocess_task.outputs['training_data_output']
    )
    evaluate_task = evaluate_model(
        model_input=train_task.outputs['model_output']
    )
    
# Compile the pipeline
compiler.Compiler().compile(
    pipeline_func=mnist_pipeline,
    package_path='../pipeline/mnist_pipeline.yaml'
)
