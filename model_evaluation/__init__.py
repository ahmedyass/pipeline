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
