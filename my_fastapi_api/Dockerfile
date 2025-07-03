FROM python:3.10-slim-buster

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR ..

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

WORKDIR my_fastapi_api/app/

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "mon_api:app", "--host", "0.0.0.0", "--port", "8000"]