# Dataset Audit

- Data root: `/home/superws/2026_Projects/FA_DCG/SAR_FEM1/data/flood_dataset`
- Total image files across train/val/test directories: 663
- Matches manuscript 663 patches: True
- Flood-event-isolated split: unknown; no event identifiers, metadata, or split manifest were found.

| Split | Exists | Images | Image size | Channels | Label values | Binary | Water pixel ratio | Missing labels |
| --- | ---: | ---: | --- | ---: | --- | ---: | ---: | --- |
| train | True | 600 | 512x512 | 3 | [0, 255] | True | 0.354441 | [] |
| val | True | 63 | 512x512 | 3 | [0, 255] | True | 0.458826 | [] |
| test | False | 0 | NA | NA | [] | False | NA | [] |
