from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.health import router as health_router
from app.api.tensors import router as tensors_router
from app.api.modules import router as modules_router
from app.api.connections import router as connections_router
from app.api.groups import router as groups_router

app = FastAPI(title="nn_vis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api/v1")
app.include_router(tensors_router, prefix="/api/v1")
app.include_router(modules_router, prefix="/api/v1")
app.include_router(connections_router, prefix="/api/v1")
app.include_router(groups_router, prefix="/api/v1")
