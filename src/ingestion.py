import pandas as pd
import numpy as np
import os

class DataIngestion:
    """
    Ladbrokes బెట్టింగ్ డేటాను సేకరించే క్లాస్.
    """
    def __init__(self, file_path):
        self.file_path = file_path

    def initiate_data_ingestion(self):
        print("Data Ingestion started...")
        try:
            # ఇక్కడ మనం డేటాను రీడ్ చేస్తున్నాం (క్లయింట్ ఇచ్చిన CSV అనుకుందాం)
            df = pd.read_csv(self.file_path)
            
            # Raw డేటా కోసం ఒక ఫోల్డర్ క్రియేట్ చేయడం
            os.makedirs('data/raw', exist_ok=True)
            df.to_csv('data/raw/betting_data.csv', index=False)
            
            print("Data Ingestion completed successfully.")
            return df
        except Exception as e:
            print(f"Error in Data Ingestion: {e}")
            return None

# టెస్టింగ్ కోసం:
# if __name__ == "__main__":
#     obj = DataIngestion("path_to_your_csv.csv")
#     data = obj.initiate_data_ingestion()