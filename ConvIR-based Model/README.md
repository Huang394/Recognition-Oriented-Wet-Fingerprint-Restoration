
# ConvIR-based Model
This folder provides the implementation of our IJCB paper:

"Recognition-Oriented Wet Fingerprint Restoration with Degradation-Aware Analysis for Small-Area Sensors"

---

## Architecture

This project is built upon the official implementation of ConvIR:

- ConvIR: Revitalizing Convolutional Network for Image Restoration  
  https://github.com/c-yn/ConvIR

Specifically, we adopt the configuration under:

- `Dehazing/ITS`

The network architecture (encoder–decoder structure, CNNBlocks, MSM, and MSA)
remains unchanged from the original ConvIR implementation.

---

## Modifications

While the backbone architecture is preserved, we adapt ConvIR to the
small-area wet fingerprint restoration task through:

- Replacing the original loss function with a hybrid **L1 + MS-SSIM** loss
- Adjusting training configurations for fingerprint image characteristics
- Constructing a paired dry–wet fingerprint training pipeline

The key contribution lies in adapting a general restoration framework 
to a recognition-oriented fingerprint restoration task with structure-aware supervision.

---

## How to Use

This implementation is based on ConvIR (Dehazing/ITS configuration). 
We retain the original training pipeline with minor adaptations for fingerprint restoration.

For detailed environment setup and training procedures, please refer to the original ConvIR repository.

https://github.com/c-yn/ConvIR/tree/main/Dehazing/ITS

---

### Pretrained Models

Pretrained models will be released upon acceptance.

Please place the downloaded weights under:
`./pretrain/`

---

## Acknowledgement

We sincerely thank the authors of ConvIR for making their code publicly available.

If you use this repository, please also cite:

```
@article{cui2024revitalizing,
  title={Revitalizing Convolutional Network for Image Restoration},
  author={Cui, Yuning and Ren, Wenqi and Cao, Xiaochun and Knoll, Alois},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
```
