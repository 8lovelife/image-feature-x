from PIL import Image
import requests
import numpy as np
from io import BytesIO
from feature_reduce_service.strategy import register_strategy
from typing import List, Union, Dict, Any
from sklearn.manifold import TSNE
from .pca import reduce_pca
import umap

@register_strategy("umap")
def reduce_umap(feature_vector: List[float], target_dim: int, intermediate: Dict[str, Any]=None, stored_features: List[List[float]] = None) -> list:
    try:
        vec = np.array(feature_vector, dtype=np.float32)
        if intermediate and 'descriptors' in intermediate:
            X = np.array(intermediate['descriptors'], dtype=np.float32)
        elif stored_features is not None and len(stored_features) >= target_dim + 1:
            X = np.vstack([np.array(stored_features, dtype=np.float32), vec.reshape(1, -1)])
        else:
            return reduce_pca(feature_vector, target_dim, intermediate, stored_features)
        umapper = umap.UMAP(n_components=target_dim)
        X_umap = umapper.fit_transform(X)
        return X_umap[-1].tolist()
    except Exception as e:
        print(f"reduce_tsne error: {e}")
        return [1.0/target_dim] * target_dim