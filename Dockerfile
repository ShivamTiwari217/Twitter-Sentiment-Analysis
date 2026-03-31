FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn[standard] torch --index-url https://download.pytorch.org/whl/cpu
COPY outputs/ outputs/
COPY model.py .
COPY fastapi_app.py .
EXPOSE 8000
CMD ["uvicorn","fastapi_app:app","--host","0.0.0.0","--port","8000"]
