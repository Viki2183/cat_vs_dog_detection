import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Set the paths
data_dir = r'C:\Users\Vrishin Dharmesh KP\Downloads\cats vs dogs\train'
categories = ['cats', 'dogs']
img_size = 224  # VGG16 expects 224x224 images


def load_images(data_dir, categories, img_size):
    data = []
    labels = []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path)
                resized_array = cv2.resize(img_array, (img_size, img_size))
                data.append(resized_array)
                labels.append(class_num)
            except Exception as e:
                pass
    return np.array(data), np.array(labels)


# Load the data
data, labels = load_images(data_dir, categories, img_size)

# Preprocess the data for VGG16
data = preprocess_input(data)

# Load VGG16 model for feature extraction
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

# Extract features
features = vgg_model.predict(data)
features = features.reshape(features.shape[0], -1)  # Flatten

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train the SVM model
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Make predictions
y_pred = svm.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


def predict_image(image_path, model, svm, img_size):
    img_array = cv2.imread(image_path)
    resized_array = cv2.resize(img_array, (img_size, img_size))
    img_data = img_to_array(resized_array)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    # Extract features
    features = model.predict(img_data)
    features = features.reshape(features.shape[0], -1)  # Flatten

    # Predict using the SVM model
    prediction = svm.predict(features)
    return categories[int(prediction[0])]


# Example usage
image_path = r'C:\Users\Vrishin Dharmesh KP\OneDrive\Documents\imageclassifysvm\animal image.webp'  # Replace with the path to your image
prediction = predict_image(image_path, vgg_model, svm, img_size)
print(f'The image is predicted to be a: {prediction}')
