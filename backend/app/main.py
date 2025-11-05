# imports
from fastapi import FastAPI
from app.core.config import settings
# from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.api.health import router as health_router
from app.api.prompt import router as prompt_router
from app.api.public import router as public_router

# now import the backend pipeline
# from app.api.factcheck import router as factcheck_router


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
# run the fact checking pipeline
# app.include_router(factcheck_router, prefix="/api")
# give the frontend the ability to change the config
app.include_router(public_router, prefix="/api")

def start_app():
    print("Environment:", settings.ENV)
    print("Using spaCy model:", settings.SPACY_MODEL)
    # print("ClaimBuster key loaded:", bool(settings.CLAIMBUSTER_API_KEY))


if __name__ == "__main__":
    start_app()
