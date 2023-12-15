import argparse
from http import HTTPStatus
import json
import time
from typing import Any, AsyncGenerator, Dict, List
from fastapi.responses import JSONResponse, Response, StreamingResponse

from fastapi import HTTPException
from schema import InferenceRequest

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

class vLLMEngine:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--host", type=str, default="0.0.0.0")
        parser.add_argument("--port", type=int, default=8000)
        parser = AsyncEngineArgs.add_cli_args(parser)
        args, _ = parser.parse_known_args()
        engine_args = AsyncEngineArgs.from_cli_args(args)
        start = time.time()
        self.async_engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.engine = self.async_engine.engine
        print("took" + " {:.2f}".format(time.time()-start) + " seconds to start vllm engine for model" + f" {args.model}")


    async def generate(
        self,
        req: InferenceRequest,
    ) -> AsyncGenerator[bytes, None] | Dict[str, List[str]]:
        req_dict = await req.json()
        prompts = req_dict.pop("prompts")
        stream = req_dict.pop("stream", False)
        sampling_params = SamplingParams(**req_dict)

        if len(prompt) != 1 and stream:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="requires exactly one prompt for streaming",
            )

        # Streaming
        if stream:
            request_id = random_uuid()
            results_generator = self.async_engine.generate(
                prompt=prompts[0],
                sampling_params=sampling_params,
                request_id=request_id,
                prompt_token_ids=None,
            )
            async def stream_results() -> AsyncGenerator[bytes, None]:
                full_output = ""
                async for request_output in results_generator:
                    text_outputs = []
                    finish_reason = ""
                    for output in request_output.outputs:
                        text_outputs.append(output.text[len(full_output):])
                        full_output += output.text[len(full_output):]
                        finish_reason = output.finish_reason
                    ret = {
                        "id": request_id,
                        "created": int(time.time()),
                        "text": text_outputs,
                        "finish_reason": finish_reason,
                    }
                    yield (json.dumps(ret) + "\n")

            return StreamingResponse(stream_results())

        # Non-streaming
        outputs = []
        for prompt in prompts:
            request_id = random_uuid()
            self.engine.add_request(str(request_id), prompt, sampling_params)
        
        while self.engine.has_unfinished_requests():
            request_outputs = self.engine.step()
            for request_output in request_outputs:
                if request_output.finished:
                    outputs.append(request_output.outputs[0].text)

        return JSONResponse({"text": outputs})
