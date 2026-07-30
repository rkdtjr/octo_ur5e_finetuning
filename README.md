# Octo UR5e Fine-tuning

Pipeline:
1. Freedrive demonstration recording
2. Joint trajectory replay
3. Replay-time synchronized image/state recording
4. Episode quality evaluation
5. Action generation
6. RLDS conversion
7. Docker-based Octo fine-tuning
8. Offline validation
9. UR5e deployment

## Collector v2 quick start

The operator enables freedrive externally. The collector never manages power,
brakes, PolyScope programs, freedrive, or controller switching.

```bash
python -m pip install -e .
octo-collector doctor --config collector/config/collector.yaml
octo-collector record-demo --config collector/config/collector.yaml --instruction "pick up the blue object"
octo-collector validate-demo data/raw/<episode_id>
octo-collector replay data/raw/<episode_id>
```

All commands are safe/dry-run by default. `record-demo --execute` enables only
keyboard commands to the configured digital output (DO0 on this hardware);
only `replay --execute` may command arm motion. Read
`docs/collector_spec.md` before hardware use.
