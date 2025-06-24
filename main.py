from fastapi import FastAPI
from api.routes import router as api_router
import feature_extract_service.extractors
import feature_reduce_service.reducers

app = FastAPI()
app.include_router(api_router)