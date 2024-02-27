import os
import kfp
from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics, ClassificationMetrics

@component(base_image="tensorflow/tensorflow", packages_to_install=['scikit-learn'])
def model_training(
    ml_model : Input[Model],
    x_train_processed : Input[Dataset], x_test_processed: Input[Dataset],
    y_train_artifact : Input[Dataset], y_test_artifact :Input[Dataset],
    hyperparameters : dict, 
    metrics: Output[Metrics], classification_metrics: Output[ClassificationMetrics], model_trained: Output[Model]
    ):
    """
    Build the model with Keras API
    Export model metrics
    """
    from tensorflow import keras
    import tensorflow as tf
    import numpy as np
    import os
    import glob
    from sklearn.metrics import confusion_matrix
    
    #load dataset
    x_train = np.load(x_train_processed.path)
    x_test = np.load(x_test_processed.path)
    y_train = np.load(y_train_artifact.path)
    y_test = np.load(y_test_artifact.path)
    
    #load model structure
    model = keras.models.load_model(ml_model.path)
    
    #reading best hyperparameters from katib
    lr=float(hyperparameters["lr"])
    no_epochs = int(hyperparameters["num_epochs"])
    
    #compile the model - we want to have a binary outcome
    model.compile(tf.keras.optimizers.SGD(learning_rate=lr),
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

    
    #fit the model and return the history while training
    history = model.fit(
      x=x_train,
      y=y_train,
      epochs=no_epochs,
      batch_size=20,
    )

     
    # Test the model against the test dataset
    # Returns the loss value & metrics values for the model in test mode.
    model_loss, model_accuracy = model.evaluate(x=x_test,y=y_test)
    
    #build a confusione matrix
    y_predict = model.predict(x=x_test)
    y_predict = np.argmax(y_predict, axis=1)
    cmatrix = confusion_matrix(y_test, y_predict)
    cmatrix = cmatrix.tolist()
    numbers_list = ['0','1','2','3','4','5','6','7','8','9']
    #log confusione matrix
    classification_metrics.log_confusion_matrix(numbers_list,cmatrix)
  
    #Kubeflox metrics export
    metrics.log_metric("Test loss", model_loss)
    metrics.log_metric("Test accuracy", model_accuracy)
    
    #adding /1/ subfolder for TFServing and saving model to artifact store
    model_trained.uri = model_trained.uri + '/1/'
    keras.models.save_model(model,model_trained.path)