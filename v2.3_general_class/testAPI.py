from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import os

app = FastAPI()

class ProductDescription(BaseModel):
    description: str

# Load the Keras model with absolute path
model_path = os.path.join(os.path.dirname(__file__), "model-neural-net.keras")
model = tf.keras.models.load_model(model_path)

@app.post("/predict/")
async def predict_category(item: ProductDescription):
    # Preprocess the input description
    input_data = preprocess_description(item.description)
    
    # Perform prediction
    prediction = model.predict(input_data)
    
    # Postprocess the output to get the category
    category = postprocess_prediction(prediction)
    
    return {"category": category}

def preprocess_description(description: str):
    # Add your preprocessing steps here
    return description

def postprocess_prediction(prediction):
    # Add your postprocessing steps here
    return prediction
