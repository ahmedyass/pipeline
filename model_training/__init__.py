from kfp.dsl import component, Input, Output, Model, Dataset

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
