import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
import pandas as pd
import numpy as np

dataset_path = 'NDVI'
classes = ['no', 'mild', 'strong']
label_dict = {'no': 0, 'mild': 1, 'strong': 2}

images_data = []
labels = []

for label in classes:
    class_path = os.path.join(dataset_path, label)
    for image_file in os.listdir(class_path):
        img_path = os.path.join(class_path, image_file)
        img = Image.open(img_path)
        img = img.resize((512, 512))
        img_array = np.array(img)

        images_data.append(img_array)
        labels.append(label_dict[label])

images_data = np.array(images_data)
labels = np.array(labels)

print(f"Loaded {len(images_data)} images with labels")

X = np.array(images_data)
y = np.array(labels)

X = X / 255.0
X = X.reshape(X.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVC(C=10, kernel='rbf', gamma='scale')

svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Test Accuracy: {accuracy}")
print(f"Test Precision: {precision}")
print(f"Test Recall: {recall}")
print(f"Test F1 Score: {f1}")
