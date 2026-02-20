import joblib
from sklearn.metrics import confusion_matrix
import pandas as pd

class ModelEvaluation:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        cm = confusion_matrix(y_test, predictions)
        
        # ఇక్కడ మనం ఒక చిన్న లాజిక్ రాస్తాం
        # ఒకవేళ Accuracy 80% కంటే తక్కువ ఉంటే మోడల్ ఫెయిల్ అయినట్లు
        print(f"Confusion Matrix:\n{cm}")
        return cm