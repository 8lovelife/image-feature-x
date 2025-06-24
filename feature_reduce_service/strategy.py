from typing import Callable, Dict
from feature_reduce_service.types import ReducerType

# Strategy function signature
FeatureReducer = Callable[[str,int], list]  # Accept image_url, return feature vector

# Registry to hold strategies
strategy_registry: Dict[ReducerType, FeatureReducer] = {}

def register_strategy(name: ReducerType):
    def wrapper(func: FeatureReducer):
        print(f"Registering feature reducer strategy: {name}")
        strategy_registry[name] = func
        return func
    return wrapper

def get_reducer(name: ReducerType) -> FeatureReducer:
    if name not in strategy_registry:
        raise ValueError(f"No feature strategy registered for: {name}")
    return strategy_registry[name]