from pydantic import BaseModel


class InferenceRequest(BaseModel):
  task_id: int

class CustomInferenceRequest(InferenceRequest):
  b64: str
