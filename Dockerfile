FROM python:3.9.16

LABEL authors="pitrLabs"

ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY Pipfile /app/

RUN pip install pipenv && pipenv --python 3.9.16

RUN pipenv install

ENV PATH="/app/.venv/bin:$PATH"

COPY . /app/
