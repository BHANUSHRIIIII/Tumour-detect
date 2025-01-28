import os
import numpy as np
import matplotlib.pyplot as plt
from cv2 import imread
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, MobileNetV2, VGG19
from keras.layers import Flatten
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# Path to the dataset
path = "brain_tumor_dataset"

# Image data generator
image_generator = ImageDataGenerator(
    rescale=1./255,
   )

# Training and testing sets
training_set = image_generator.flow_from_directory(batch_size=16,
                                                   directory=path,
                                                   shuffle=True,
                                                   target_size=(224, 224),
                                                   subset="training",
                                                   color_mode='rgb',
                                                   class_mode='binary')

testing_set = image_generator.flow_from_directory(batch_size=16,
                                                  directory=path,
                                                  shuffle=False,
                                                  target_size=(224, 224),
                                                  subset="validation",
                                                  color_mode='rgb',
                                                  class_mode='binary')

# Load the pre-trained models without their top layers (feature extraction only)
resnet_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
mobilenet_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
vgg_model = VGG19(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze the layers of the models
for model in [resnet_model, mobilenet_model, vgg_model]:
    for layer in model.layers:
        layer.trainable = False

# Helper function to extract features
def extract_features(model, data):
    features = []
    labels = []
    for images, label in data:
        feature = model.predict(images)  # This uses batch processing
        features.append(feature.reshape(images.shape[0], -1))  # Reshape and append features
        labels.append(label)
    return np.vstack(features), np.hstack(labels)

# Extract features from ResNet50, MobileNetV2, and VGG19
resnet_features, labels = extract_features(resnet_model, training_set)
mobilenet_features, _ = extract_features(mobilenet_model, training_set)
vgg_features, _ = extract_features(vgg_model, training_set)

# Combine the features from all three models
combined_features = np.hstack([resnet_features, mobilenet_features, vgg_features])

# Split the combined features into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.1, random_state=42)

# Train a Random Forest classifier on the combined features
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the Random Forest model
y_pred = rf.predict(X_test)

# Confusion matrix and classification report
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot ROC curve
y_pred_proba = rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve')
plt.show()

# Feature extraction and evaluation on the test set
resnet_test_features, test_labels = extract_features(resnet_model, testing_set)
mobilenet_test_features, _ = extract_features(mobilenet_model, testing_set)
vgg_test_features, _ = extract_features(vgg_model, testing_set)

# Combine the test features
combined_test_features = np.hstack([resnet_test_features, mobilenet_test_features, vgg_test_features])

# Evaluate on the test set
test_pred = rf.predict(combined_test_features)
print("Test Confusion Matrix:\n", confusion_matrix(test_labels, test_pred))
print("\nTest Classification Report:\n", classification_report(test_labels, test_pred))
