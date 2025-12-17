from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO: WHAT IS THE EXPECTED SIZE????
TARGET_SIZE = (128, 128)
CLASS_NAMES = [
    'buffalo', 'capybara', 'cat', 'deer', 'dog', 'elephant', 'giraffe',
    'jaguar', 'kangaroo', 'lion', 'parrot', 'penguin', 'rhino', 'sheep',
    'tiger', 'turtle', 'zebra'
]
CONFIDENCE_THRESHOLD = 0.95

# model = None
# @app.on_event("startup")
# async def load_model_on_startup():
#     """Loads the model once when the API starts."""
#     global model
#     model = load_model('animal_classifier.h5')
#     print("Model loaded successfully!")


def preprocess_image(image_data: bytes) -> np.ndarray:
    """Preprocesses the image for model prediction."""
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize(TARGET_SIZE)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


'''
Function to handle prediction and send it back to frontend
'''
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess image
    image_data = await file.read()
    image_array = preprocess_image(image_data)

    model = load_model('animal_classifier.h5')
    predictions = model.predict(image_array)[0]
    
    # Get max confidence and corresponding class
    max_confidence = np.max(predictions)
    predicted_class_idx = np.argmax(predictions)

    # Return 'Unknown' if confidence is below threshold
    if max_confidence < CONFIDENCE_THRESHOLD:
        predicted_class = "Unknown"
        confidence = float(max_confidence)
    else:
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(max_confidence)
    
    # Create response with all class probabilities
    all_predictions = {
        CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))
    }
    
    print(f"Predicted: {predicted_class} with confidence {confidence}")

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_predictions": all_predictions,
        "threshold_used": CONFIDENCE_THRESHOLD
    }


@app.get("/")
async def root():
    return {
        "message": "ZooSnap Animal Classification API",
        "classes": CLASS_NAMES,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }

