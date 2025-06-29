from PIL import Image
import requests
import numpy as np
from io import BytesIO
from feature_reduce_service.strategy import register_strategy
from typing import List, Union, Dict, Any
from sklearn.decomposition import PCA

@register_strategy("pca")
def reduce_pca(feature_vector: List[float], target_dim: int, intermediate: Dict[str, Any]=None, stored_features: List[List[float]] = None) -> list:
    try:
        vec = np.array(feature_vector, dtype=np.float32)
        if intermediate and 'descriptors' in intermediate:
            X = np.array(intermediate['descriptors'], dtype=np.float32)
        elif stored_features is not None and len(stored_features) >= target_dim + 1:
            X = np.vstack([np.array(stored_features, dtype=np.float32), vec.reshape(1, -1)])
        else:
            return _variance_based_selection(vec, target_dim)
        pca = PCA(n_components=target_dim)
        X_pca = pca.fit_transform(X)
        reduced = X_pca[-1]
        norm = np.linalg.norm(reduced)
        if norm > 0:
            reduced = reduced / norm
        return reduced.tolist()
    except Exception as e:
        print(f"reduce_pca error: {e}")
        raise RuntimeError(f"reduce_pca error: {e}")
        return [1.0/target_dim] * target_dim

def _variance_based_selection(feature_vector: np.ndarray, target_dim: int) -> List[float]:
    importance_scores = np.abs(feature_vector)
    
    positions = np.arange(len(feature_vector))
    center = len(feature_vector) / 2
    position_weights = 1.0 / (1.0 + np.abs(positions - center) / center * 0.1)
    
    final_scores = importance_scores * position_weights
    
    top_indices = np.argpartition(final_scores, -target_dim)[-target_dim:]
    top_indices = np.sort(top_indices) 
    
    result = feature_vector[top_indices]
    
    if result.sum() > 0:
        result = result / result.sum()
    else:
        result = np.ones(target_dim) / target_dim
    
    return result.tolist()        