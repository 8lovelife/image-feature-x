
import time
from typing import List

import numpy as np
from api.types import DimReductionVectors, FinalResult
from feature_extract_service.types import FeatureType
from feature_extract_service.strategy import get_extractor
from feature_reduce_service.strategy import get_reducer
import asyncio

async def run_feature_pipeline(extract: str,imageUrl: str,dimensions: int) -> FinalResult:
    extractor = get_extractor(extract)
    loop = asyncio.get_running_loop()
    start_time = time.time();
    if asyncio.iscoroutinefunction(extractor):
        extract_result = await extractor(imageUrl, dimensions)
    else:
        extract_result = await loop.run_in_executor(
            None, extractor, imageUrl, dimensions
        )
    end_time = time.time();
    elapsed = end_time - start_time;
    print(f"[extractor {extract}] Execution time: {elapsed:.4f} seconds")

    if hasattr(extract_result, "original_vector"):
        original_vector = extract_result.original_vector
        intermediate = getattr(extract_result, "intermediate", None)
    else:
        original_vector = extract_result
        intermediate = None

    # if not isinstance(original_vector, list):
    #     original_vector = list(original_vector)
    final = FinalResult(original_vector=original_vector)

    algos = ["pca", "umap", "tsne"]
    # tasks_to_gather: Dict[str, asyncio.Future] = {}
    if intermediate and 'descriptors' in intermediate:
        stored_features = intermediate['descriptors']
    else:
        stored_features = generate_fake_samples(original_vector).tolist()

    for algo in algos:
        start_time = time.time();
        reducer = get_reducer(algo)
        
        if asyncio.iscoroutinefunction(reducer):
            vec1_result = await reducer(original_vector, 1, intermediate, stored_features)
            vec2_result = await reducer(original_vector, 2, intermediate, stored_features)
            vec3_result = await reducer(original_vector, 3, intermediate, stored_features)
        else:
            loop = asyncio.get_event_loop()
            vec1_result = await loop.run_in_executor(None, reducer, original_vector, 1, intermediate, stored_features)
            vec2_result = await loop.run_in_executor(None, reducer, original_vector, 2, intermediate, stored_features)
            vec3_result = await loop.run_in_executor(None, reducer, original_vector, 3, intermediate, stored_features)
        
        setattr(final, algo, DimReductionVectors(
            vector_1d=list(vec1_result),
            vector_2d=list(vec2_result),
            vector_3d=list(vec3_result),
        ))
        end_time = time.time();
        elapsed = end_time - start_time;
        print(f"[reducer {algo}] Execution time: {elapsed:.4f} seconds")
        print(f"algo {algo} done")


    return final


def generate_fake_samples(
    vec: List[float],
    num_fake_samples: int = 30,
    noise_std: float = 0.01
) -> np.ndarray:
    base = np.array(vec, dtype=np.float32)
    fake_samples = [base + np.random.normal(0, noise_std, size=base.shape)
                    for _ in range(num_fake_samples)]
    return np.vstack([base] + fake_samples)