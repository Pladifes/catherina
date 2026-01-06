# Pixi Docker Compose Workflow for Interactive Development

This guide shows how to build and run your Pixi project in a Docker container using **Docker Compose**, keeping the container alive for interactive use with bind-mounted input/output folders.

---

## 1. Create `docker-compose.yml`

Place this in your project root:

```yaml
version: "3.9"

services:
  app:
    build: .
    image: pixi-test:latest
    volumes:
      - ./data/input:/app/data/input:ro
      - ./data/output:/app/data/output
    command: tail -f /dev/null
```

## 2. Build and start the container
```bash
docker compose up --build -d
```

## 3. Find container id
```bash
docker ps
```

## 4. Open an interactive shell
```bash
docker compose exec container-id bash
```

## 5. Run a script
```bash
pixi run synth
```

## 6. Stop the container
```bash
docker compose down
```