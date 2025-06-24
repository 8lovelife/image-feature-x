from typing import Literal,List, Dict, Any, Optional
from pydantic import BaseModel

FeatureType = Literal[
    "resnet", "vgg", "mobilenet", "sift", "hog", "lbp", "color_histogram", "orb"
]

class ExtractResult(BaseModel):
    original_vector: List[float]
    intermediate: Optional[Dict[str, Any]] = None