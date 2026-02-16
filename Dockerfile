FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy project metadata first (for layer caching)
COPY pyproject.toml ./

# Install core + graph deps via uv
RUN uv pip install --system --no-cache ".[graph,cloud]"

# Install PyG packages from CUDA 12.4 wheel index
RUN uv pip install --system --no-cache \
    torch-geometric \
    torch-sparse \
    torch-scatter \
    --extra-index-url https://data.pyg.org/whl/torch-2.5.0+cu124.html

# Install transformers (for SapBERT embeddings)
RUN uv pip install --system --no-cache "transformers>=4.35"

# Pre-download SapBERT model at build time
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext'); \
    AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')"

# Copy source code and config
COPY src/ src/
COPY config/ config/
COPY data/mappings/ data/mappings/
COPY ontology/ ontology/

ENTRYPOINT ["python", "-m", "src.cloud_train"]
