
FROM python:3.11-slim

WORKDIR /app

# System deps for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Optional extras: astropy; uncomment if desired
# RUN pip install --no-cache-dir astropy

COPY . .

ENV AL_STORE_DIR=/app/store
ENV AL_DB_PATH=/app/store/app.db

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
