import base64
from io import BytesIO
import json
from pathlib import Path

import cv2
from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image

from api.inference import run_inference
from api.schemas.schema import InferenceRequest, CustomInferenceRequest


ABS = Path(__file__).resolve().parents[2]

router = APIRouter(prefix='/api')

@router.post(path='/transition')
async def transition(req: InferenceRequest):
  task_id = req.task_id
  with open(ABS / 'transition.json') as f:
    d = json.load(f)

  return JSONResponse({
    'transition': d[str(task_id)]
  }, status_code=200)

@router.post(path='/inference')
async def inference(req: InferenceRequest):
  prob = run_inference(req.task_id)

  return JSONResponse({
    'prob': prob
  }, status_code=200)

@router.post(path='/inference/custom')
async def custom_inference(req: CustomInferenceRequest):
  image_data = base64.b64decode(req.b64)
  buffer = BytesIO(image_data)
  image = Image.open(buffer).convert('RGB')  # RGBに変換しておくのが無難

  prob = run_inference(req.task_id, image)

  return JSONResponse({
    'prob': prob
  }, status_code=200)
