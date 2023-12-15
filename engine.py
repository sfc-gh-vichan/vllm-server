import argparse
from dataclasses import dataclass
from http import HTTPStatus
import json
import os
from pathlib import Path
import time
from typing import AsyncGenerator, Dict, List
from dacite import from_dict
from fastapi.responses import JSONResponse, StreamingResponse

from args import args
from fastapi import HTTPException
from schema import InferenceRequest
from sse import EventType, event

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid


@dataclass
class DeploymentConfig:
    model_name: str
    tensor_parallel: int
    max_model_len: int
    truncate: bool


class vLLMEngine:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--host", type=str, default="0.0.0.0")
        parser.add_argument("--port", type=int, default=8000)
        parser = AsyncEngineArgs.add_cli_args(parser)
        args, _ = parser.parse_known_args()
        engine_args = AsyncEngineArgs.from_cli_args(args)
        engine_args.tensor_parallel_size = self.tensor_parallel
        engine_args.max_model_len = self.max_model_len
        start = time.time()
        self.async_engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.engine = self.async_engine.engine
        print("took" + " {:.2f}".format(time.time()-start) + " seconds to start vllm engine for model" + f" {args.model}")


    def _load_deployment_config(self):
        path = args.model_repository
        p = Path(path)
        model_path = ""
        for f in p.iterdir():
            if f.is_dir():
                model_path = f.name
                break
        if len(model_path) == 0:
            raise ValueError()
        deployment_config_file_path = os.path.join(path, model_path, "deployment_config.json")
        f = open(deployment_config_file_path)
        config = json.load(f)
        deployment_config = from_dict(data_class=DeploymentConfig, data=config)
        self.model_name = deployment_config.model_name
        self.tensor_parallel = deployment_config.tensor_parallel
        self.max_model_len = deployment_config.max_model_len
        self.truncate = deployment_config.truncate


    def list_models(self):
        return JSONResponse({"model": self.model_name})


    async def generate(
        self,
        req: InferenceRequest,
    ) -> AsyncGenerator[bytes, None] | Dict[str, List[str]]:
        req_dict = req.dict()
        prompts = req_dict.pop("prompts")
        stream = req_dict.pop("stream", False)
        sampling_params = SamplingParams(**req_dict)

        if len(prompts) != 1 and stream:
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
                    message = {
                        "id": request_id,
                        "created": int(time.time()),
                        "text": text_outputs,
                        "finish_reason": finish_reason,
                    }
                    yield event(event_type=EventType.MESSAGE, message=message)
                yield event(event_type=EventType.DONE, message="")

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

        return JSONResponse({"texts": outputs})
