FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir -e '.[llm]'

EXPOSE 8000
# offline demo by default; mount a .env / set USE_REAL_PROVIDERS=true for live data
ENV USE_REAL_PROVIDERS=false
CMD ["uvicorn", "wealthwise.app:app", "--host", "0.0.0.0", "--port", "8000"]
