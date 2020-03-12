FROM python:3.7

ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code
COPY . /code/
RUN pip install -r requirements.txt
CMD gunicorn mlTest.wsgi:application --bind 0.0.0.0:$PORT