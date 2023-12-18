# https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers
FROM us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-1:latest
COPY requirements.txt ./src/*.py ./
RUN python -m pip install --upgrade pip -r requirements.txt 

