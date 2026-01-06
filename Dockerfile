# Use a lightweight Linux base image
FROM debian:12

# Install curl and certificates (needed to install Pixi)
RUN apt-get update && apt-get install -y curl ca-certificates && update-ca-certificates

# Install Pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash

# Make Pixi available in PATH
ENV PATH="/root/.pixi/bin:${PATH}"

# Set working directory inside the container
WORKDIR /app

# Copy your project files into the container
COPY . .

# Install dependencies from pixi.toml
RUN pixi install

# Default CMD: pass any argument to pixi run
ENTRYPOINT ["pixi", "run"]
