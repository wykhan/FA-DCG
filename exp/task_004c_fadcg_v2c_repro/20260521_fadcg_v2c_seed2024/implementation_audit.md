# Implementation Audit

## FA-DCG V2c

- Backbone-aware selected configuration from task_004a/task_004b.
- FCN uses the V2a conservative SAR frequency gate because it was best for FCN.
- U-Net uses V2b bottleneck dilation bias plus a low-strength V2a skip calibration block.
- Skip calibration alpha is initialized to 0.25 to limit early perturbation.

## Unit Test Output

```text
task_004c FA-DCG V2c tests passed
```
