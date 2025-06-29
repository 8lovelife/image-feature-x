import time
from fastapi import APIRouter, Query
from typing import Literal
from core_service.feature_pipeline import run_feature_pipeline
from feature_extract_service.types import FeatureType
from feature_extract_service.strategy import get_extractor
from feature_reduce_service.strategy import get_reducer
from .types import FinalResult,DimReductionVectors
import asyncio

router = APIRouter()

@router.get("/feature-pipeline", response_model=FinalResult)
async def feature_pipeline_api(
    extract: FeatureType = Query(..., alias="extract"),
    imageUrl: str = Query(..., alias="imageUrl"),
    dimensions: int = Query(...,alias="dimensions")
):
    return await run_feature_pipeline(extract, imageUrl, dimensions)
    
    # extractor = get_extractor(extract)
    # loop = asyncio.get_running_loop()
    # start_time = time.time();
    # if asyncio.iscoroutinefunction(extractor):
    #     extract_result = await extractor(imageUrl, dimensions)
    # else:
    #     extract_result = await loop.run_in_executor(
    #         None, extractor, imageUrl, dimensions
    #     )
    # end_time = time.time();
    # elapsed = end_time - start_time;
    # print(f"[extractor {extract}] Execution time: {elapsed:.4f} seconds")

    # if hasattr(extract_result, "original_vector"):
    #     original_vector = extract_result.original_vector
    #     intermediate = getattr(extract_result, "intermediate", None)
    # else:
    #     original_vector = extract_result
    #     intermediate = None

    # # if not isinstance(original_vector, list):
    # #     original_vector = list(original_vector)
    # final = FinalResult(original_vector=original_vector)

    # algos = ["pca", "umap", "tsne"]
    # # tasks_to_gather: Dict[str, asyncio.Future] = {}

    # for algo in algos:
    #     start_time = time.time();
    #     reducer = get_reducer(algo)
        
    #     if asyncio.iscoroutinefunction(reducer):
    #         vec1_result = await reducer(original_vector, 1, intermediate, None)
    #         vec2_result = await reducer(original_vector, 2, intermediate, None)
    #         vec3_result = await reducer(original_vector, 3, intermediate, None)
    #     else:
    #         loop = asyncio.get_event_loop()
    #         vec1_result = await loop.run_in_executor(None, reducer, original_vector, 1, intermediate, None)
    #         vec2_result = await loop.run_in_executor(None, reducer, original_vector, 2, intermediate, None)
    #         vec3_result = await loop.run_in_executor(None, reducer, original_vector, 3, intermediate, None)
        
    #     setattr(final, algo, DimReductionVectors(
    #         vector_1d=list(vec1_result),
    #         vector_2d=list(vec2_result),
    #         vector_3d=list(vec3_result),
    #     ))
    #     end_time = time.time();
    #     elapsed = end_time - start_time;
    #     print(f"[reducer {algo}] Execution time: {elapsed:.4f} seconds")
    #     print(f"algo {algo} done")


    # return final


@router.get("/test", response_model=FinalResult)
def test():
    return {
        "original_vector": [0.1, 0.2, 0.3],
        "pca": {
            "vector_1d": [0.1],
            "vector_2d": [0.1, 0.2],
            "vector_3d": [0.1, 0.2, 0.3]
        },
        "umap": None,
        "tsne": None
    }

@router.get("/feature-vector")
async def generate_feature_vector_api(
    featureType: FeatureType = Query(..., alias="featureType"),
    imageUrl: str = Query(..., alias="imageUrl"),
    dimensions: int = Query(...,alias="dimensions")
):
    extractor = get_extractor(featureType)
    vector = extractor(imageUrl,dimensions)

    reducer = get_reducer("pca")

    async def reduce_wrapper(target_dim):
        loop = asyncio.get_running_loop()
        return await reducer(vector, target_dim)
        # if reducer is a sync fun，use run_in_executor wrapper
        # return await loop.run_in_executor(None, reducer, vector, target_dim)

    tasks = [
        reduce_wrapper(3),
        reduce_wrapper(2),
        reduce_wrapper(1),
    ]
    vector_3d, vector_2d, vector_1d = await asyncio.gather(*tasks)

    return {"vector": vector,"3d_vector":vector_3d,"2d_vector":vector_2d,"1d_vector":vector_1d}


@router.get("/feature-pipeline2", response_model=FinalResult)
async def feature_pipeline_api(
    extract: FeatureType = Query(..., alias="extract"),
    imageUrl: str = Query(..., alias="imageUrl"),
    dimensions: int = Query(..., alias="dimensions"),
):
    extractor = get_extractor(extract)
    loop = asyncio.get_running_loop()
    if asyncio.iscoroutinefunction(extractor):
        extract_result = await extractor(imageUrl, dimensions)
    else:
        extract_result = await loop.run_in_executor(
            None, extractor, imageUrl, dimensions
        )

    if hasattr(extract_result, "original_vector"):
        original_vector = extract_result.original_vector
        intermediate = getattr(extract_result, "intermediate", None)
    else:
        original_vector = extract_result
        intermediate = None

    final = FinalResult(original_vector=list(original_vector))
    algos = ["pca"]
    
    # 1. Create a list of all coroutines that need to be run
    all_coroutines = []
    for algo in algos:
        reducer = get_reducer(algo)
        for dim in [1, 2, 3]:
            if asyncio.iscoroutinefunction(reducer):
                coro = reducer(original_vector, dim, intermediate, None)
            else:
                coro = loop.run_in_executor(None, reducer, original_vector, dim, intermediate, None)
            all_coroutines.append(coro)

    # 2. Run all 9 operations concurrently
    all_results = await asyncio.gather(*all_coroutines)

    # 3. Assign the results to the final object
    result_index = 0
    for algo in algos:
        # The results are in a flat list, ordered by algo then dimension
        vec1 = list(all_results[result_index])
        vec2 = list(all_results[result_index + 1])
        vec3 = list(all_results[result_index + 2])
        result_index += 3

        setattr(final, algo, DimReductionVectors(
            vector_1d=vec1,
            vector_2d=vec2,
            vector_3d=vec3,
        ))

    return final