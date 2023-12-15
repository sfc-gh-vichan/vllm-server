from pydantic import BaseModel
from typing import List


class InferenceRequest(BaseModel):
    prompts: List[str]
    stream: bool = False
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
