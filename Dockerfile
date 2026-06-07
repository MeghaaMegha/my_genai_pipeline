# Start directly from the official public proxy mirror to bypass CDN blocks
FROM mirror.gcr.io/library/ubuntu:latest

WORKDIR /app

# Point configuration links directly to your system's path binaries
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV PYTHONPATH="/app"

# Copy your local configurations and source code files into the container workspace
COPY pyproject.toml logging.conf ./
COPY src/ ./src/

# Instruct the container to fire using a direct, universal path execution link
CMD ["/usr/bin/python3", "-m", "src.main"]
