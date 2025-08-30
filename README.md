# Folder: upskillCampus
# ├── Code/
# │   ├── data_preprocessing.py
# │   ├── model_training.py
# │   ├── disease_prediction.py
# │   └── irrigation_control.ino
# └── Report/
#     └── Agriculture_Project_Report.pdf


# -----------------------------
# data_preprocessing.py
# -----------------------------
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image preprocessing for training and testing
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('dataset/train',
                                              target_size=(128,128),
                                              batch_size=32,
                                              class_mode='categorical')

test_set = test_datagen.flow_from_directory('dataset/test',
                                            target_size=(128,128),
                                            batch_size=32,
                                            class_mode='categorical')

# -----------------------------
# model_training.py
# -----------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# CNN model for crop disease classification
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(128,128,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=4, activation='softmax'))  # Example: 4 crop classes

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_set,
                    validation_data=test_set,
                    epochs=10)

# Save the trained model
model.save("crop_disease_model.h5")

# -----------------------------
# disease_prediction.py
# -----------------------------
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model("crop_disease_model.h5")

def predict_crop(img_path):
    img = image.load_img(img_path, target_size=(128,128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    classes = ['Healthy', 'Leaf Rust', 'Powdery Mildew', 'Bacterial Blight']
    prediction = model.predict(x)
    print("Prediction:", classes[np.argmax(prediction)])

# Example usage
# predict_crop("sample_leaf.jpg")

# -----------------------------
# irrigation_control.ino (Arduino Code)
# -----------------------------
'''
#define sensorPin A0
#define motorPin 7

int moisture_level;

void setup() {
  pinMode(motorPin, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  moisture_level = analogRead(sensorPin);
  Serial.println(moisture_level);

  if (moisture_level < 400) {   // threshold value
    digitalWrite(motorPin, HIGH);  // turn pump ON
    Serial.println("Soil dry: Pump ON");
  } else {
    digitalWrite(motorPin, LOW);   // turn pump OFF
    Serial.println("Soil wet: Pump OFF");
  }

  delay(1000);
}
'''
