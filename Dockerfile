FROM tensorflow/tensorflow:1.4.0-gpu-py3
USER root

# Install build utilities
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

# Check our python environment
RUN python3 --version
RUN pip3 --version

# Set the working directory
WORKDIR /model

# Installing python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the files from the project’s root to the working directory
COPY ./ /model/
RUN ls -la /model/*

# Define default command.
CMD ["bash"]