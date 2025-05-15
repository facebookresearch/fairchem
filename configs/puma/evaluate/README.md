Running PUMA Evaluations
------------------------

Conserving val and test
```bash
fairchem -c configs/puma/evaluate/puma_conserving_val.yaml cluster=h100 checkpoint=puma_sm
fairchem -c configs/puma/evaluate/puma_conserving_val.yaml cluster=h100 checkpoint=puma_sm
```

Direct val and test
```bash
fairchem -c configs/puma/evaluate/puma_conserving_val.yaml cluster=h100 checkpoint=puma_lg
fairchem -c configs/puma/evaluate/puma_conserving_val.yaml cluster=h100 checkpoint=puma_lg
```
