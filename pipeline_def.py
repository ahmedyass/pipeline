from kfp.dsl import pipeline
import kfp.compiler as compiler
from data_processing import preprocess_data
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
