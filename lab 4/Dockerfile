FROM python:3

WORKDIR /src

RUN pip install flask
RUN pip install flask-mysql

COPY . /src

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

CMD ["flask", "run"]