import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Model and image path configuration
img_width, img_height = 64, 64
model_path = 'traffic_light_recognition_model.h5'
image_path = 'image.jpg'  # can be .jpg, .png, etc.

# Load the model
model = load_model(model_path)

# Function to preprocess the image
def preprocess_image(image_path, target_size):
    image = load_img(image_path, target_size=target_size, color_mode='rgb')  # Ensuring RGB
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0  # Normalizing to [0, 1]
    return image

# Prepare the image
image = preprocess_image(image_path, (img_width, img_height))

# Make prediction
predictions = model.predict(image)
class_index = np.argmax(predictions, axis=1)

# Convert index to class
classes = ['Green', 'Red', 'Yellow']
predicted_class = classes[class_index[0]]

print(f"The traffic light is: {predicted_class}")
