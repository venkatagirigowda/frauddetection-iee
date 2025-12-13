# Use a Python slim image for a smaller final image size
FROM python:3.12-slim

# Define the working directory inside the container
WORKDIR /app

# --- 1. Define AWS build arguments (These are SECRET) ---
# Docker will accept these keys only for the build process, preventing them from leaking
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

# --- 2. Install System Dependencies ---
RUN apt-get update -y && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    python3-dev \
    # Clean up apt caches to reduce image size
    && rm -rf /var/lib/apt/lists/*

# --- 3. Install Python Dependencies ---
# Copy requirements.txt first to leverage Docker layer caching
COPY requirements.txt .

RUN pip install --upgrade pip
# This step installs dvc[s3] and all other required libraries
RUN pip install --no-cache-dir -r requirements.txt

# --- 4. Copy Application Files and DVC config/placeholders ---
# Copy the DVC configuration files (.dvc folder) and your application structure
# (e.g., fastapisetup/app.py, artifacts/*.dvc files)
COPY . .

# --- 5. DVC PULL STEP (The Fix for 'Artifacts not ready') ---
# This runs only during the build and downloads the actual model files from S3.
RUN if [ -f .dvc/config ]; then \
    echo "DVC config found. Attempting to pull artifacts from S3..."; \
    # Set the environment variables ONLY for the duration of the dvc pull command
    AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID_S3 \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY_S3 \
    dvc pull; \
    echo "DVC pull complete. Artifacts should now be in /app/artifacts"; \
    fi

# --- 6. Configuration ---
# Expose the port for Uvicorn
EXPOSE 8000

# --- 7. Run the Application ---
# The CMD defines the entry point for the container
CMD ["python", "-m", "uvicorn", "fastapisetup.app:app", "--host", "0.0.0.0", "--port", "8000"]