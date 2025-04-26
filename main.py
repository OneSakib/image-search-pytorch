import numpy as np
import cv2
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.neighbors import NearestNeighbors

# Load Pre-trained Model
base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
model = Model(inputs=base_model.input, outputs=base_model.output)


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def extract_features(img_path):
    img = preprocess_image(img_path)
    features = model.predict(img, verbose=0)
    features = features.flatten()
    features = features / np.linalg.norm(features)  # normalize
    return features


# 1. Load Dataset
dataset_folder = "dataset_images/"  # folder where your images are stored
image_paths = [os.path.join(dataset_folder, img) for img in os.listdir(dataset_folder)]

# 2. Extract Features for all Images
feature_list = []
for img_path in image_paths:
    feature = extract_features(img_path)
    feature_list.append(feature)

feature_list = np.array(feature_list)

# 3. Fit Nearest Neighbors
neighbors = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="cosine")
neighbors.fit(feature_list)


# 4. Search Function
def search_similar_images(query_img_path):
    query_feature = extract_features(query_img_path)
    distances, indices = neighbors.kneighbors([query_feature])

    print("\nTop 5 similar images:")
    for idx in indices[0]:
        print(image_paths[idx])


# Example usage
query_image = "query.jpeg"  # image you want to search
search_similar_images(query_image)
