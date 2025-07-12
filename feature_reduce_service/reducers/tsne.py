from PIL import Image
import requests
import numpy as np
from io import BytesIO
from feature_reduce_service.strategy import register_strategy
from typing import List, Union, Dict, Any
from sklearn.manifold import TSNE
from .pca import reduce_pca

@register_strategy("tsne")
def reduce_tsne(feature_vector: List[float], target_dim: int, intermediate: Dict[str, Any]=None, stored_features: List[List[float]] = None) -> list:
    try:
        if target_dim not in (2, 3, 1):
            raise ValueError("t-SNE usually only supports reduction to 1, 2, or 3 dimensions")
        vec = np.array(feature_vector, dtype=np.float32)
        if intermediate and 'descriptors' in intermediate:
            X = np.array(intermediate['descriptors'], dtype=np.float32)
        elif stored_features is not None and len(stored_features) >= target_dim + 1:
            X = np.vstack([np.array(stored_features, dtype=np.float32), vec.reshape(1, -1)])
        else:
            return reduce_pca(feature_vector, target_dim, intermediate, stored_features)
        tsne = TSNE(n_components=target_dim)
        X_emb = tsne.fit_transform(X)
        return X_emb[-1].tolist()
    except Exception as e:
        print(f"reduce_tsne error: {e}")
        raise RuntimeError(f"reduce_tsne error: {e}")