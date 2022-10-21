FROM python:3.7

WORKDIR /deepclaim-ws

COPY ./requirements.txt /deepclaim-ws/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /deepclaim-ws/requirements.txt

COPY ./app /deepclaim-ws/app
COPY ./models /deepclaim-ws/models

COPY ./deepclaim_ws /deepclaim-ws/deepclaim_ws

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
