from typing import Callable, Dict
from feature_extract_service.types import FeatureType,ExtractResult

# Strategy function signature
FeatureExtractor = Callable[[str,int], ExtractResult]  # Accept image_url, return feature vector

# Registry to hold strategies
strategy_registry: Dict[FeatureType, FeatureExtractor] = {}

def register_strategy(name: FeatureType):
    def wrapper(func: FeatureExtractor):
        print(f"Registering feature extractor strategy: {name}")
        strategy_registry[name] = func
        return func
    return wrapper

def get_extractor(name: FeatureType) -> FeatureExtractor:
    if name not in strategy_registry:
        raise ValueError(f"No feature strategy registered for: {name}")
    return strategy_registry[name]