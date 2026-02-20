from src.ingestion import DataIngestion
from src.preprocessing import DataTransformation
from src.training import ModelTrainer
import pandas as pd

def main():
    # 1. Data Ingestion (డేటా సేకరణ)
    # గమనిక: మీ దగ్గర డేటా ఫైల్ ఉంటే దాని పాత్ ఇక్కడ ఇవ్వండి
    data_path = "data/raw_betting_data.csv" 
    ingestor = DataIngestion(data_path)
    raw_df = ingestor.initiate_data_ingestion()

    if raw_df is not None:
        # 2. Data Transformation (ఫీచర్ ఇంజనీరింగ్)
        transformer = DataTransformation()
        processed_df = transformer.transform_data(raw_df)

        # 3. Model Training & MLflow Tracking (ట్రైనింగ్)
        # processed_df లో 'is_fraud' అనే టార్గెట్ కాలమ్ ఉందని నిర్ధారించుకోండి
        trainer = ModelTrainer(processed_df)
        model = trainer.train_model()
        
        print("--- MLOps Pipeline Executed Successfully ---")
    else:
        print("Pipeline failed at Ingestion stage.")

if __name__ == "__main__":
    main()