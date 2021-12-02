FROM python:3.9
ENV HOST 0.0.0.0
ENV PORT 8080
ENV APP_MODULE app.api:app
ENV NB_WORKERS 2

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./config.json /code/config.json

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN apt-get update -y

RUN apt-get install gunicorn uvicorn -y

COPY ./app /code/app

# CMD ["uvicorn", APP_MODULE, "--host", HOST, "--port", PORT]

CMD gunicorn $APP_MODULE -w $NB_WORKERS -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8888 --timeout 60