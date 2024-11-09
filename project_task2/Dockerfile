FROM python:3.10.12

WORKDIR /project

ADD requirements.txt .
ADD main.py .

COPY ./ml ./ml
COPY ./templates ./templates

RUN pip install -r requirements.txt

CMD [ "python", "./main.py" ]