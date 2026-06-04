#!/usr/bin/env bash
set -euo pipefail

check() {
  local name="$1"
  local url="$2"
  echo -n "Aguardando ${name}... "
  for i in {1..60}; do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      echo "ok"
      return 0
    fi
    sleep 2
  done
  echo "TIMEOUT"
  return 1
}

check "MinIO"       "http://localhost:9000/minio/health/live"
check "MLflow"      "http://localhost:5000/health"
check "API"         "http://localhost:8000/health"
check "Prometheus"  "http://localhost:9090/-/ready"
check "Grafana"     "http://localhost:3000/api/health"

echo "✓ Todos os serviços responderam."
