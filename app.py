from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
from catboost import CatBoostClassifier
# Charger le modèle sauvegardé

MODEL_PATH = "catboost_model.pkl"


try:
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    print("Modèle chargé avec succès via CatBoost.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {str(e)}")

# Initialiser l'application FastAPI
app = FastAPI()

# Définir un schéma de requête avec Pydantic
class PredictionInput(BaseModel):
    features: list[float]  # Liste des caractéristiques en entrée

# Définir la route pour la prédiction
@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Convertir les données en format numpy
        input_array = np.array(input_data.features).reshape(1, -1)
        # Effectuer la prédiction
        prediction = model.predict(input_array)
        # Retourner le résultat sous format JSON
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction : {str(e)}")

