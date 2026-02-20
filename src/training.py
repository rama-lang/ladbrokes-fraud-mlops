import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
import joblib
import os

class ModelTrainer:
    def __init__(self, df):
        self.df = df

    def train_model(self):
        print("Model Training started...")
        
        # 1. Features మరియు Target ని వేరు చేయడం
        # 'is_fraud' అనేది మనం కనిపెట్టాల్సిన టార్గెట్ కాలమ్ అనుకుందాం
        X = self.df.drop(['is_fraud', 'timestamp', 'user_id'], axis=1)
        y = self.df['is_fraud']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 2. MLflow Tracking ప్రారంభం
        with mlflow.start_run():
            # మోడల్ క్రియేషన్
            rf = RandomForestClassifier(n_estimators=100, max_depth=10)
            rf.fit(X_train, y_train)

            # Predictions
            y_pred = rf.predict(X_test)

            # Metrics లెక్కించడం
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # MLflow లో డేటా సేవ్ చేయడం (ఇది క్లైంట్ కి రిపోర్ట్ లాగా ఉపయోగపడుతుంది)
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("f1_score", f1)

            # మోడల్ ని రిజిస్ట్రీ లో సేవ్ చేయడం
            mlflow.sklearn.log_model(rf, "fraud_model")

            # లోకల్ గా కూడా మోడల్ సేవ్ చేయడం
            os.makedirs('models', exist_ok=True)
            joblib.dump(rf, 'models/betting_fraud_model.pkl')
            
            print(f"Training Complete! Recall: {recall:.2f}, F1: {f1:.2f}")
            return rf