from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse
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
    resp = await engine.generate(req)
    if req.stream:
        return StreamingResponse(resp)
    return JSONResponse(resp)
