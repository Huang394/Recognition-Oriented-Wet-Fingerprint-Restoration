# Recognition Oriented Wet Fingerprint Restoration
This repository provides the implementation of our IJCB paper:

"Recognition-Oriented Wet Fingerprint Restoration with Degradation-Aware Analysis for Small-Area Sensors"

This work focuses on wet fingerprint restoration from a recognition-oriented perspective, 
aiming to preserve ridge structures that are critical for reliable matching.
---

## Pipeline
The proposed framework consists of three components:

1. Fingerprint restoration model (released in this repository)
2. Wet quality classifier
3. Recognition evaluation (TAR@FAR, FRR)

Due to licensing and data restrictions, components (2) and (3) are not included in this anonymous version.
They will be released upon acceptance.

## Model Details

The fingerprint restoration model is implemented based on ConvIR with task-specific adaptations.

For detailed architecture and training configurations, please refer to:
`ConvIR-based Model/README.md`

## Anonymous Submission

This repository is provided for anonymous review. 
Some components are withheld due to licensing and anonymization constraints, 
and will be released upon acceptance.
