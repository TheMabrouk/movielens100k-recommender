FROM python:3.10.5

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

CMD ["sudo", "uvicorn", "app.main:app",  "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]
