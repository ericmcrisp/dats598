# imports
from fastapi import FastAPI
from config import settings
from api.health import router as health_router
from api.prompt import router as prompt_router

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello World", "env": settings.ENV}


# health check route
app.include_router(health_router, prefix="/api")
# prompt string GET
app.include_router(prompt_router, prefix="/api")


def start_app():
    print("Environment:", settings.ENV)
    print("Using spaCy model:", settings.SPACY_MODEL)
    print("ClaimBuster key loaded:", bool(settings.CLAIMBUSTER_API_KEY))


if __name__ == "__main__":
    start_app()
