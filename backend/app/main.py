# imports
from fastapi import FastAPI
from app.core.config import settings
# from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.api.health import router as health_router
from app.api.prompt import router as prompt_router
from app.api.public import router as public_router
from app.api.factcheck import router as factcheck_router


app = FastAPI()
# because Vite and backend on different ports
app.add_middleware(
    CORSMiddleware,
    # might need to change this for prod
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Hello World", "env": settings.ENV}


# health check route
app.include_router(health_router, prefix="/api")
# prompt string GET
app.include_router(prompt_router, prefix="/api")
# give the frontend the ability to change the config
app.include_router(public_router, prefix="/api")
# run the fact checking pipeline
app.include_router(factcheck_router, prefix="/api")


def start_app():
    print("=" * 50)
    print(f"Environment: {settings.ENV}")
    print(f"Using spaCy model: {settings.SPACY_MODEL}")
    print(f"Frontend URL: {settings.FRONTEND_URL}")
    print(f"Backend Port: {settings.BACKEND_PORT}")
    print("=" * 50)
    
    # Preload ML models if needed
    # try:
    #     from app.api.ml import get_model
    #     print("Loading SentenceTransformer model...")
    #     get_model()
    #     print("✓ ML model loaded successfully")
    # except Exception as e:
    #     print(f"⚠️  Warning: Could not preload ML model: {e}")


if __name__ == "__main__":
    start_app()
