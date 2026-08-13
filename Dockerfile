FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies first (layer caching: only rebuilds on requirements change)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code (data/ is bind-mounted at runtime, see docker-compose.yml)
COPY app.py ./
COPY pages ./pages
COPY src ./src

EXPOSE 8501

# Bind 0.0.0.0 so the container is reachable from the host
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
