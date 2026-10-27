# syntax=docker/dockerfile:1
# AML-GraphMiner image: core library + REST API + web interface + plotting (CPU).
#
#   docker build -f docker/Dockerfile -t aml-graphminer .                       # light image
#   docker build -f docker/Dockerfile -t aml-graphminer:gnn --build-arg WITH_GNN=1 .   # + PyTorch/PyG (CPU)
#   docker run --rm -p 127.0.0.1:8000:8000 aml-graphminer                       # demo at http://127.0.0.1:8000/ui/
#
# For GPU training start from an NVIDIA/PyTorch base image instead and install
# "aml-graphminer[gnn]" plus the pyg-lib wheel matching its torch / CUDA build.
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip wheel --no-deps --wheel-dir /wheels .

FROM python:3.12-slim
ARG WITH_GNN=0
ARG TORCH_VERSION=2.10.0
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl
COPY --from=build /wheels /wheels
RUN if [ "$WITH_GNN" = "1" ]; then \
        pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu && \
        pip install pyg-lib -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html" && \
        pip install "$(ls /wheels/aml_graphminer-*.whl)[api,viz,gnn,explain]" ; \
    else \
        pip install "$(ls /wheels/aml_graphminer-*.whl)[api,viz]" ; \
    fi && rm -rf /wheels
RUN useradd --create-home --uid 10001 amlgm
USER amlgm
WORKDIR /home/amlgm
EXPOSE 8000
# default: the web interface on a synthetic demo (a demo detector is trained at start-up)
CMD ["amlgm", "serve", "--demo", "--host", "0.0.0.0", "--port", "8000"]
