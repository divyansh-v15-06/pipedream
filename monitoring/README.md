# Pipedream training monitor

Start the local dashboard from the repository root:

```bash
python3 monitoring/server.py
```

Open <http://127.0.0.1:8765>. The monitor reads the `pipedream-overnight`
container, TensorBoard event files, checkpoints, Docker stats, and host GPU
telemetry. It is read-only and does not control or restart training.

Environment overrides:

```bash
PIPEDREAM_CONTAINER=my-run \
PIPEDREAM_OUTPUT_DIR=results/raw/my-run/ppo \
PIPEDREAM_TOTAL_TIMESTEPS=100000 \
python3 monitoring/server.py
```
