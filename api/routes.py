from fastapi import APIRouter, Query
from core_service.feature_pipeline import run_feature_pipeline
from feature_extract_service.types import FeatureType
from .types import FinalResult

router = APIRouter()

@router.get("/feature-pipeline", response_model=FinalResult)
async def feature_pipeline_api(
    extract: FeatureType = Query(..., alias="extract"),
    imageUrl: str = Query(..., alias="imageUrl"),
    dimensions: int = Query(...,alias="dimensions")
):
    return await run_feature_pipeline(extract, imageUrl, dimensions)