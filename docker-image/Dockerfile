# Define function directory
ARG FUNCTION_DIR="/function"

FROM python:3.8-buster as build-image

# Install aws-lambda-cpp build dependencies
RUN apt-get update && \
  apt-get install -y \
  g++ \
  make \
  cmake \
  unzip \
  libcurl4-openssl-dev

# Include global arg in this stage of the build
ARG FUNCTION_DIR
# Create function directory
RUN mkdir -p ${FUNCTION_DIR}

# Install the runtime interface client
RUN pip install \
        --target ${FUNCTION_DIR} \
        awslambdaric

COPY yolo-requirements.txt .
RUN pip install \
        --target ${FUNCTION_DIR} \
        -r yolo-requirements.txt

COPY requirements.txt .
RUN pip install \
        --target ${FUNCTION_DIR} \
        -r requirements.txt

# Copy function code
COPY app/ ${FUNCTION_DIR}

# Multi-stage build: grab a fresh copy of the base image
FROM python:3.8-buster

ENV FUNCTION_DIR ${FUNCTION_DIR}

# Include global arg in this stage of the build
ARG FUNCTION_DIR
# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}

# Copy in the build image dependencies
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "app.handler" ]