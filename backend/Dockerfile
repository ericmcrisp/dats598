FROM python:3.10

WORKDIR /app

COPY requirements-core.txt .

RUN pip install --no-cache-dir -r requirements-core.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
