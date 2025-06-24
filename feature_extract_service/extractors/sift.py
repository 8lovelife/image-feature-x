import numpy as np
import cv2
from sklearn.decomposition import PCA
import requests
from PIL import Image
from io import BytesIO
from feature_extract_service.strategy import register_strategy
from feature_extract_service.types import ExtractResult


def load_image_from_url(image_url: str) -> np.ndarray:
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('L')
        return np.array(img)
    except Exception as e:
        raise Exception(f"Failed to load image from URL: {e}")

def compute_sift_features(image_array: np.ndarray) -> tuple:
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image_array, None)
    return keypoints, descriptors

def reduce_dimensions(descriptors: np.ndarray, dimension: int) -> np.ndarray:
    if descriptors is None or len(descriptors) == 0:
        raise ValueError("No SIFT descriptors found in the image.")
    
    if dimension >= descriptors.shape[1]:
        return descriptors
    
    pca = PCA(n_components=dimension)
    reduced = pca.fit_transform(descriptors)
    return reduced

@register_strategy("sift")
async def extract_sift(image_url: str,dm: int) -> ExtractResult:
    try:
        image_array = load_image_from_url(image_url)
        
        keypoints, descriptors = compute_sift_features(image_array)
        
        if descriptors is None or len(descriptors) == 0:
            return [0.0] * dm
        
        if dm < descriptors.shape[1]:
            reduced_features = reduce_dimensions(descriptors, dm)
        else:
            reduced_features = descriptors
        
        if len(reduced_features) > 1:
            global_feature = np.mean(reduced_features, axis=0)
        else:
            global_feature = reduced_features[0]
        
        if len(global_feature) != dm:
            if len(global_feature) < dm:
                padded_feature = np.zeros(dm)
                padded_feature[:len(global_feature)] = global_feature
                global_feature = padded_feature
            else:
                global_feature = global_feature[:dm]
        
        norm = np.linalg.norm(global_feature)
        if norm > 0:
            global_feature = global_feature / norm
        
        return ExtractResult(
            original_vector=global_feature.tolist(),
            intermediate={
                "descriptors": descriptors.tolist() if descriptors is not None else []
            }
        )
        
    except Exception as e:
        print(f"SIFT extract error: {e}")
        return ExtractResult(
            original_vector=[0.0] * dm,
        )