FROM ghcr.io/astral-sh/uv:debian-slim
WORKDIR /app
COPY pyproject.toml uv.lock /app/
RUN uv sync --locked
COPY . /app/
EXPOSE 5000
CMD ["uv", "run", "app.py", "dev", "--host", "0.0.0.0", "--port", "5000"]