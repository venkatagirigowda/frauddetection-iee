FROM python:3.12-slim

WORKDIR /app

RUN apt-get update -y && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install dvc[s3]

COPY fastapisetup /app/api_service
COPY artifacts /app/model_artifacts
COPY . .

EXPOSE 8000

CMD ["sh","-c","dvc pull && python -m uvicorn fastapisetup.app:app --host 0.0.0.0 --port 8000"]