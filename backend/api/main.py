from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers import router


ABS = Path(__file__).resolve().parent

app = FastAPI()
app.include_router(router)

origins = ['http://localhost:5173']
app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credenrials=True,
  allow_methods=['*'],
  allow_headers=['*']
)

@app.exception_handler(RequestValidationError)
async def handler(request: Request, exc: RequestValidationError):
  print(exc)
  return JSONResponse(content={}, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)
