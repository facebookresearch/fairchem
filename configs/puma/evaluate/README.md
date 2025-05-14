Running PUMA Evaluations
------------------------

Conserving val and test
```bash
fairchem -c configs/puma/evaluate/puma_conserving.yaml cluster=h100 checkpoint=puma_sm
```

Direct val and test
```bash
fairchem -c configs/puma/evaluate/puma_direct.yaml cluster=h100 checkpoint=puma_lg
```
