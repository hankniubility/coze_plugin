from fastapi import FastAPI

from .pareto_plugin import router as pareto_router

app = FastAPI(title="Pareto Multi-Objective Optimizer")
app.include_router(pareto_router)

