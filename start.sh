#!/bin/bash
PORT=${PORT:-4000}
uvicorn main:app --host 0.0.0.0 --port $PORT
