# Implementation Audit

## task_005c Placement Design

- No new FA-DCG mechanism is introduced.
- V1.1 uses `VectorizedLightFADC`; V2d uses `FADCGV2d`.
- One block is inserted per model to isolate placement.
- FCN placements: after conv3 and after conv4.
- U-Net placements: after encoder stage 2 and after encoder stage 3.
- Weak residual initialization: `alpha_init=0.25`.

## Unit Test Output

```text
fcn v1.1 after_conv3: params=134.3076M, output=(2, 1, 256, 256)
fcn v2d after_conv4: params=134.4323M, output=(2, 1, 256, 256)
unet v1.1 after_encoder2: params=31.0520M, output=(2, 1, 256, 256)
unet v2d after_encoder3: params=31.0898M, output=(2, 1, 256, 256)
```
