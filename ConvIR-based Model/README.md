
# Recognition-Oriented-Wet-Fingerprint-Restoration

This repository provides the implementation of our IJCB paper:

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

The key contribution lies in **task adaptation and structure-sensitive supervision**,  
rather than architectural redesign.

---

## How to Use

The training and testing procedures are identical to the official ConvIR
implementation under the `Dehazing/ITS` configuration:

https://github.com/c-yn/ConvIR/tree/main/Dehazing/ITS

Please follow the original repository for:

- Environment setup  
- Dataset preparation  
- Training instructions  
- Testing and evaluation procedures  

---

### Pretrained Models

Our pretrained restoration models will provided on Google Drive upon acceptance, please put the pretrained model under: ```./pretrain/```

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
