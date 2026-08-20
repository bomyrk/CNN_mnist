"""
FastAPI serving application for Fashion-MNIST CNN model.
Exposes endpoints for health checks and image classification predictions.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow import keras
import os

#app = FastAPI(title="Fashion-MNIST CNN API")

# Fashion-MNIST class labels
CLASS_LABELS = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# Model path
MODEL_PATH = "models/on_the_go/fashion_mnist_api_final.keras"

# Load model at startup
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the CNN model on startup."""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    yield  # App runs here

app = FastAPI(title="Fashion-MNIST CNN API", lifespan=lifespan)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess uploaded image to match model input requirements.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Preprocessed numpy array with shape (1, 28, 28, 1)
    """
    # Load image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 28x28 if needed
    if image.size != (28, 28):
        image = image.resize((28, 28), Image.LANCZOS)
    
    # Convert to numpy array and normalize
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    # Reshape to (1, 28, 28, 1) for model input
    image_array = image_array.reshape(1, 28, 28, 1)
    
    return image_array


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict the class of an uploaded Fashion-MNIST image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON response with predicted class, label, and confidence score
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        
        return {
            "predicted_class": predicted_class,
            "predicted_label": CLASS_LABELS[predicted_class],
            "confidence": confidence,
            "all_probabilities": {
                CLASS_LABELS[i]: float(predictions[0][i]) 
                for i in range(len(CLASS_LABELS))
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_array")
async def predict_array(data: dict):
    """
    Predict from a JSON array of pixel values (28x28 grayscale).
    
    Args:
        data: JSON with 'pixels' key containing 784 values (0-255)
        
    Returns:
        JSON response with predicted class, label, and confidence score
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        pixels = data.get("pixels")
        if not pixels or len(pixels) != 784:
            raise HTTPException(
                status_code=400, 
                detail="Input must contain 784 pixel values (28x28)"
            )
        
        # Convert to numpy array and reshape
        image_array = np.array(pixels, dtype=np.float32).reshape(1, 28, 28, 1) / 255.0
        
        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        
        return {
            "predicted_class": predicted_class,
            "predicted_label": CLASS_LABELS[predicted_class],
            "confidence": confidence,
            "all_probabilities": {
                CLASS_LABELS[i]: float(predictions[0][i]) 
                for i in range(len(CLASS_LABELS))
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


#if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run(app, host="0.0.0.0", port=8000)


