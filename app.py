from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from http import HTTPStatus

from engine import vLLMEngine
from schema import InferenceRequest


engine = vLLMEngine()
app = FastAPI()


@app.get("/model")
async def _model() -> Response:
    return engine.list_model()


@app.get("/health")
async def _health() -> Response:
    return Response(status_code=HTTPStatus.OK)


@app.post("/v1/generate")
async def _generate(req: InferenceRequest) -> Response:
    return await engine.generate(req)
