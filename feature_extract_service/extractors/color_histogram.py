from PIL import Image
import requests
import numpy as np
from io import BytesIO
from feature_extract_service.strategy import register_strategy
from feature_extract_service.types import ExtractResult


@register_strategy("color_histogram")
async def extract_color_histogram(image_url: str,dm: int = 256) -> ExtractResult:
    try:
        response = requests.get(image_url, timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        r, g, b = image.split()
        bins_per_channel = dm // 3
        hist_r = np.histogram(np.array(r).flatten(), bins=bins_per_channel, range=(0, 256))[0]
        hist_g = np.histogram(np.array(g).flatten(), bins=bins_per_channel, range=(0, 256))[0]
        hist_b = np.histogram(np.array(b).flatten(), bins=bins_per_channel, range=(0, 256))[0]
        hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)

        if hist.sum() > 0:
            hist /= hist.sum()
        else:
            hist = np.ones(len(hist)) / len(hist)

        return ExtractResult(
            original_vector=hist.tolist(),
            intermediate={
                "hist_r": hist_r.tolist(),
                "hist_g": hist_g.tolist(),
                "hist_b": hist_b.tolist(),
            }
        )
    except Exception as e:
        print(f"COLOR_HISTOGRAM extract error: {e}")
        raise RuntimeError(f"COLOR_HISTOGRAM extract error: {e}")
        return ExtractResult(
            original_vector=[1.0/dm] * dm,
        )