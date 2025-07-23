# Start from a ubuntu image
FROM docker.io/library/ubuntu:24.04 

# Remove the interation with the shell
ENV DEBIAN_FRONTEND=noninteractive

# Install some useful packages on ubuntu
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libfftw3-dev \
    libgsl-dev \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Create a python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create a working directory for ExSHalos
WORKDIR /ExSHalos

# Install the python requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install ExSHalos
COPY . . 
RUN pip3 install .

# Set the working directory
WORKDIR /Workspace

# Create an user for using the container 
RUN useradd --create-home --shell /bin/bash appuser
USER appuser
