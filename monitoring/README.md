# GenRecall Monitoring

一键启动 Prometheus、Grafana 和 NVIDIA DCGM Exporter：

```bash
cd /home/c00913906/genrecall/learn-llm/monitoring
docker compose up -d
```

访问入口：

- Grafana: <http://127.0.0.1:3000>
- Prometheus: <http://127.0.0.1:9090>
- DCGM metrics: <http://127.0.0.1:9400/metrics>
- vLLM metrics: <http://127.0.0.1:8000/metrics>

Grafana 默认账号密码：

```text
admin / admin
```

可以通过环境变量覆盖：

```bash
GRAFANA_ADMIN_USER=admin GRAFANA_ADMIN_PASSWORD='your-password' docker compose up -d
```

当前 compose 使用 `network_mode: host`。这样 Prometheus 配置里的
`127.0.0.1:8000` 可以直接抓到宿主机上启动的 vLLM server。

已内置 Grafana dashboard：

- `GenRecall vLLM + GPU Observability`

覆盖的核心观察项包括 TTFT、ITL/TPOT、E2E latency、P50/P95/P99、
prompt/output tokens per second、running/waiting requests、KV cache usage、
prefix cache queries/hits、GPU compute、显存和功耗。

如果之前手动启动过同名或占用端口的容器，先停止旧容器，例如：

```bash
docker stop prometheus grafana dcgm-exporter 2>/dev/null || true
docker stop genrecall-prometheus genrecall-grafana genrecall-dcgm-exporter 2>/dev/null || true
```
