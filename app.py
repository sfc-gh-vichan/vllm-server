from fastapi import FastAPI
from fastapi.responses import Response
from http import HTTPStatus

from engine import vLLMEngine
from schema import InferenceRequest


engine = vLLMEngine()
app = FastAPI()


@app.get("/health")
async def health() -> Response:
    return Response(status_code=HTTPStatus.OK)


@app.post("/v1/generate")
async def _generate(req: InferenceRequest) -> Response:
    return await engine.generate(req)
