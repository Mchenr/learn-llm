#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-prometheus}"
PROMETHEUS_IMAGE="${PROMETHEUS_IMAGE:-prom/prometheus}"
PROMETHEUS_CONFIG="${PROMETHEUS_CONFIG:-/home/c00913906/genrecall/prometheus/prometheus.yml}"
PROMETHEUS_DATA_DIR="${PROMETHEUS_DATA_DIR:-/home/c00913906/genrecall/prometheus/data}"
PROMETHEUS_RETENTION="${PROMETHEUS_RETENTION:-15d}"

if ! command -v docker >/dev/null 2>&1; then
  printf 'Error: docker command not found.\n' >&2
  exit 1
fi

if [[ ! -f "$PROMETHEUS_CONFIG" ]]; then
  printf 'Error: Prometheus config does not exist: %s\n' "$PROMETHEUS_CONFIG" >&2
  exit 1
fi

mkdir -p "$PROMETHEUS_DATA_DIR"

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  NETWORK_MODE="$(docker inspect -f '{{.HostConfig.NetworkMode}}' "$CONTAINER_NAME")"
  if [[ "$NETWORK_MODE" != "host" ]]; then
    printf 'Error: existing container %s uses network mode %s, but host mode is required\n' \
      "$CONTAINER_NAME" "$NETWORK_MODE" >&2
    printf 'because prometheus.yml targets 127.0.0.1:8000.\n' >&2
    printf 'Recreate it with:\n  docker rm -f %q\n  %q\n' \
      "$CONTAINER_NAME" "$0" >&2
    exit 1
  fi

  if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" == "true" ]]; then
    printf 'Prometheus container is already running: %s\n' "$CONTAINER_NAME"
  else
    docker start "$CONTAINER_NAME" >/dev/null
    printf 'Started existing Prometheus container: %s\n' "$CONTAINER_NAME"
  fi
else
  docker run -d \
    --name "$CONTAINER_NAME" \
    --network host \
    -v "$PROMETHEUS_CONFIG:/etc/prometheus/prometheus.yml:ro" \
    -v "$PROMETHEUS_DATA_DIR:/prometheus" \
    "$PROMETHEUS_IMAGE" \
    --config.file=/etc/prometheus/prometheus.yml \
    --storage.tsdb.retention.time="$PROMETHEUS_RETENTION" >/dev/null
  printf 'Created and started Prometheus container: %s\n' "$CONTAINER_NAME"
fi

printf 'Prometheus UI: http://127.0.0.1:9090\n'
printf 'vLLM target:   http://127.0.0.1:8000/metrics\n'
printf 'Prometheus data: %s (retention: %s)\n' \
  "$PROMETHEUS_DATA_DIR" "$PROMETHEUS_RETENTION"
