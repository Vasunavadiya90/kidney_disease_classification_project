import os
import tensorflow as tf
from pathlib import Path
from urllib.parse import urlparse
import mlflow
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories,save_json
from dotenv import load_dotenv

load_dotenv()

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        # Use DagsHub tracking (already initialized in main.py)
        # Set or create experiment
        mlflow.set_experiment("kidney_disease_classification")
        
        try:
            with mlflow.start_run():
                mlflow.log_params(self.config.all_params)
                mlflow.log_metrics(
                    {"loss": self.score[0], "accuracy": self.score[1]}
                )
                # Log model artifacts
                mlflow.keras.log_model(self.model, "model")
                print("URI:", os.getenv("MLFLOW_TRACKING_URI"))
                print("User:", os.getenv("MLFLOW_TRACKING_USERNAME"))
                print("Tracking URI:", mlflow.get_tracking_uri())
                print("MLflow logging successful!")
                
        except Exception as e:
            print(f"Warning: MLflow logging failed: {e}")
            print("Model evaluation completed, but MLflow tracking is unavailable.")