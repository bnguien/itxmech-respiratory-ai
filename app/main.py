from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.lung_sound import router as lung_sound_router
from app.core.config import get_settings
from app.pipelines.lung_pipeline import LungSoundPipeline

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.lung_pipeline = LungSoundPipeline()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.include_router(lung_sound_router, prefix=settings.api_prefix)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "respicare-ai",
    }
