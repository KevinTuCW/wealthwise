"""Wealthwise FastAPI application — Phase 1 skeleton.

Only /health is exposed here; workbench and advisory routes come in Phase 3.
"""
from fastapi import FastAPI

app = FastAPI(
    title="WealthWise",
    description="金融投顾多 Agent 系统 (阵03)",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
