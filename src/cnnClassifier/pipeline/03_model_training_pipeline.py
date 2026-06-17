from cnnClassifier.components.model_training import Training
from cnnClassifier.config.configuration import ConfigurationManager
import os




try:
    # Navigate to project root
    from pathlib import Path
    project_root = Path(r"e:\data science\Deep_learning_project\kidney_disease_classification")
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}\n")
    
    config = ConfigurationManager()
    training_config = config.get_training_config()
    print(f"Training config loaded")
    print(f"Training data: {training_config.training_data}")
    print(f"Base model: {training_config.updated_base_model_path}\n")
    
    training = Training(config=training_config)
    training.get_base_model()
    print(f"Base model loaded\n")
    
    training.train_valid_generator()
    print(f"Data generators created\n")
    
    training.train()
    print(f"\n✓ Training completed successfully!")
    
except Exception as e:
    print(f"✗ Training failed: {str(e)}")
    import traceback
    traceback.print_exc()
    raise e
