FROM nvidia/cuda:latest
USER root

# Install build utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.6 \
    python3-pip \
    python3-setuptools \
    && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

# Check our python environment
RUN python3 --version
RUN pip3 --version

# Set the working directory
WORKDIR /model

# Installing python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy all the files from the project’s root to the working directory
COPY ./ /model/
RUN ls -la /model/*

# Define default command.
CMD ["bash"]
