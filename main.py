# This is my main.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
from predictor import get_latest_prediction
import uvicorn
import os
import sys  # Import sys to handle paths
import signal
import logging
import logging.config
from logging.handlers import RotatingFileHandler

# --- This is the key change ---
# Determine if we are running in a PyInstaller bundle
# This helps in creating the correct path for the log file
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the sys.executable is the path to the exe
    application_path = os.path.dirname(sys.executable)
else:
    # If run as a script, the path is the script's directory
    application_path = os.path.dirname(os.path.abspath(__file__))

# Create a logs directory if it doesn't exist
log_dir = os.path.join(application_path, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'app.log')

# --- Robust Logging Configuration for Packaged Apps ---
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(asctime)s - %(message)s",
            "use_colors": False, # Important: Disable colors for file logging
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            "use_colors": False, # Important: Disable colors
        },
    },
    "handlers": {
        # This handler writes logs to the 'app.log' file
        "file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "default",
            "filename": log_file_path,
            "maxBytes": 1024 * 1024 * 5,  # 5 MB
            "backupCount": 3,
            "encoding": "utf8",
        },
        # This handler is for access logs, also writing to the same file
        "access_file_handler": {
             "class": "logging.handlers.RotatingFileHandler",
            "formatter": "access",
            "filename": log_file_path,
            "maxBytes": 1024 * 1024 * 5,  # 5 MB
            "backupCount": 3,
            "encoding": "utf8",
        },
    },
    "loggers": {
        "uvicorn": {
            "level": "INFO",
            "handlers": ["file_handler"],
            "propagate": False,
        },
        "uvicorn.error": {
            "level": "INFO",
            "handlers": ["file_handler"],
            "propagate": False,
        },
        "uvicorn.access": {
            "level": "INFO",
            "handlers": ["access_file_handler"],
            "propagate": False,
        },
    },
}


app = FastAPI()

# Allow CORS...
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Determine frontend path correctly for both script and bundled app
frontend_path = os.path.join(application_path, "frontend.html")

@app.get("/", response_class=HTMLResponse)
async def read_frontend():
    with open(frontend_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/predict")
async def predict():
    try:
        # Log the request
        logging.getLogger("uvicorn.error").info("Prediction requested.")
        result = get_latest_prediction()
        return result
    except Exception as e:
        # Log the error
        logging.getLogger("uvicorn.error").exception("An error occurred during prediction.")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/shutdown")
async def shutdown():
    logging.getLogger("uvicorn.error").info("Shutdown endpoint called. Server is terminating.")
    os.kill(os.getpid(), signal.SIGTERM)
    return JSONResponse(content={"message": "Server is shutting down..."})

if __name__ == "__main__":
    uvicorn.run(
        app,  # Pass the app object directly!
        host="0.0.0.0",
        port=8000,
        log_config=LOGGING_CONFIG
    )
