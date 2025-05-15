Running UMA Evaluations
------------------------

Conserving val and test
```bash
fairchem -c configs/uma/evaluate/uma_conserving.yaml cluster=h100 checkpoint=uma_sm
```

Direct val and test
```bash
fairchem -c configs/uma/evaluate/uma_direct.yaml cluster=h100 checkpoint=uma_lg
```
