import argparse
from http import HTTPStatus
import json
import time
from typing import Any, AsyncGenerator, Dict, List

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
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("took" + " {:.2f}".format(time.time()-start) + " seconds to start vllm engine for model" + f" {args.model}")


    async def generate(
        self,
        req: InferenceRequest,
    ) -> AsyncGenerator[bytes, None] | Dict[str, List[str]]:
        req_dict = await req.model_dump_json()
        prompt = req_dict.pop("prompt")
        stream = req_dict.pop("stream", False)
        sampling_params = SamplingParams(**req_dict)
        request_id = random_uuid()

        # Replace this with token ids with arg prompt_token_ids=prompt_token_ids after tokenization
        results_generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            prompt_token_ids=None,
        )

        # Streaming
        if stream:
            async def stream_results() -> AsyncGenerator[bytes, None]:
                async for request_output in results_generator:
                    prompt = request_output.prompt
                    text_outputs = [
                        prompt + output.text for output in request_output.outputs
                    ]
                    ret = {"text": text_outputs}
                    yield (json.dumps(ret) + "\0").encode("utf-8")

            return stream_results()

        # Non-streaming
        final_output = None
        async for request_output in results_generator:
            if await req.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                raise HTTPException(
                    status_code=499,
                    details="client closed connection",
                )
            final_output = request_output

        prompt = final_output.prompt
        text_outputs = [prompt + output.text for output in final_output.outputs]
        return {"text": text_outputs}