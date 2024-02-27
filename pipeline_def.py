from kfp.dsl import pipeline
import kfp.compiler as compiler
from data_loading import load_dataset
from data_processing import preprocessing
from model_building import model_building
from model_training import model_training

@pipeline(
    name='mnist-classifier-dev',
    description='Detect digits'
)
def mnist_pipeline(hyperparameters: dict):
    load_task = load_dataset()
    preprocess_task = preprocessing(
        x_train_artifact = load_task.outputs["x_train_artifact"],
        x_test_artifact = load_task.outputs["x_test_artifact"]
    )

    model_building_task = model_building()

    training_task = model_training(
        ml_model = model_building_task.outputs["ml_model"],
        x_train_processed = preprocess_task.outputs["x_train_processed"],
        x_test_processed = preprocess_task.outputs["x_test_processed"],
        y_train_artifact = load_task.outputs["y_train_artifact"],
        y_test_artifact = load_task.outputs["y_test_artifact"],
        hyperparameters = hyperparameters
    )

    
# Compile the pipeline
compiler.Compiler().compile(
    pipeline_func=mnist_pipeline,
    package_path='output/mnist_pipeline.yaml'
)
