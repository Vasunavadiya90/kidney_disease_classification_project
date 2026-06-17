from cnnClassifier.pipeline.stage_01_data_ingestion_pipeline import DataIngestionTrainingPipeline
from cnnClassifier import logger
from cnnClassifier.pipeline.stage_02_prepare_base_model_pipeline import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_training_pipeline import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation_pipeline import EvaluationPipeline
import dagshub
import mlflow
import os
from dotenv import load_dotenv

load_dotenv()
# Initialize DagsHub at the very beginning
dagshub.init(repo_owner='Vasunavadiya90', repo_name='kidney_disease_classification_project', mlflow=True)
STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Prepare base model"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_base_model = PrepareBaseModelTrainingPipeline()
   prepare_base_model.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Training"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_trainer = ModelTrainingPipeline()
   model_trainer.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Evaluation stage"

try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   obj = EvaluationPipeline()
   obj.main()
   print(os.getenv("MLFLOW_TRACKING_URI"))
   print(os.getenv("MLFLOW_TRACKING_USERNAME"))
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e