from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager
from app.api_v1.auth import init_db
from app.api_v1 import endpoints
from app.models.signlens_service import service_status

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB on startup (Bypassed for offline local testing)
    # init_db()
    yield

app = FastAPI(
    title="SignLens API",
    version="2.0.0",
    lifespan=lifespan
)

# Custom OpenAPI schema to include WebSockets
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description="SignLens API with real-time sign language detection support.",
        routes=app.routes,
    )
    
    # Manually document the WebSocket endpoint
    # Note: OpenAPI doesn't natively support WebSockets, so we describe it as a GET with 101 response
    openapi_schema["paths"]["/api/v1/predict-stream"] = {
        "get": {
            "summary": "WebSocket: Real-time Prediction Stream",
            "description": (
                "Connect via `ws://` or `wss://` to stream video frames. \n\n"
                "**Usage:**\n"
                "1. Connect with query parameters `api_key` and `type`.\n"
                "2. Send raw bytes of image frames (JPEG/PNG).\n"
                "3. Receive one JSON message per frame. Show a letter when `stable` is true.\n\n"
                "*Note: This endpoint is not interactive in Swagger UI.*"
            ),
            "tags": ["Prediction"],
            "parameters": [
                {
                    "name": "api_key",
                    "in": "query",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Valid API key for authentication"
                },
                {
                    "name": "type",
                    "in": "query",
                    "required": True,
                    "schema": {"type": "string", "enum": ["video", "stream"]},
                    "description": "Type of input source"
                },
                {
                    "name": "mode",
                    "in": "query",
                    "required": False,
                    "schema": {"type": "string", "enum": ["letters", "digits"], "default": "letters"},
                    "description": "Which signs to recognise (letters and numbers are separate models)"
                }
            ],
            "responses": {
                "101": {"description": "Switching Protocols (WebSocket Handshake Success)"},
                "403": {"description": "Forbidden - Invalid API Key"},
                "400": {"description": "Bad Request - Missing parameters"}
            }
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Include versioned API router
app.include_router(endpoints.router, prefix="/api/v1")

# Health check endpoint: also reports whether the v2 models are loaded
@app.get("/", tags=["Health"])
def health_check():
    models = service_status()
    return {
        "status": "ok" if models["ready"] else "models_missing",
        "message": "SignLens API is running" if models["ready"] else models.get("error"),
        "version": "2.0.0",
        "data": {
            "Supported input types": ["image", "video", "stream"],
            "Modes": models.get("modes", []),
            "Motion signs": models.get("motion_labels", []),
            "API documentation": "/docs",
        }
    }
