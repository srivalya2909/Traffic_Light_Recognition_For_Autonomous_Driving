import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Model configuration
img_width, img_height, img_num_channels = 64, 64, 3
batch_size = 32
num_epochs = 10
num_classes = 3
loss_function = 'categorical_crossentropy'
optimizer = 'adam'
metrics = ['accuracy']
train_data_dir = 'train_dataset'  # Ensure this path is correct
validation_data_dir = 'validation_dataset'  # Ensure this path is correct

# Define the model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_width, img_height, img_num_channels)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

# Data preparation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Prepare data iterators
train_iterator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_iterator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

print("Training steps:", train_iterator.samples // batch_size)
print("Validation steps:", validation_iterator.samples // batch_size)

# Train the model
history = model.fit(
    train_iterator,
    steps_per_epoch=train_iterator.samples // batch_size,
    epochs=num_epochs,
    validation_data=validation_iterator,
    validation_steps=validation_iterator.samples // batch_size
)

# Save the model
model.save('traffic_light_recognition_model.h5')

print("Training complete. Model saved.")
