from pydantic import BaseModel
from typing import List, Optional

class DimReductionVectors(BaseModel):
    vector_1d: List[float]
    vector_2d: List[float]
    vector_3d: List[float]

class FinalResult(BaseModel):
    original_vector: List[float]
    pca: Optional[DimReductionVectors] = None
    umap: Optional[DimReductionVectors] = None
    tsne: Optional[DimReductionVectors] = None