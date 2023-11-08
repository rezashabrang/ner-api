FROM python:3.8-slim
WORKDIR /app
# set environment variables
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH="${PATH}:/root/.poetry/bin" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/phrase_api \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Tehran

# install dependencies
RUN apt update && apt upgrade -y && apt install curl -y

# POETRY
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

COPY tensorflow_gpu-2.9.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl pyproject.toml poetry.lock /app/

RUN poetry install -n --only main && rm -f ./tensorflow_gpu-2.9.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

COPY . /app

RUN rm -f ./tensorflow_gpu-2.9.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whlcommand

# Final
# --workers 6
CMD uvicorn ner_api.main:app --host 0.0.0.0 --port 80 --log-level ${LOG_LEVEL}
