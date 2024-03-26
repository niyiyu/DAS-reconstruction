# Distributed Acoustic Sensing Compression and Reconstruction using Machine Learning
![SIREN_vs_RFFN](./figures/siren_vs_rffn_50_40epoch.gif)

## Tutorials
Jupyter notebooks are provided that showcases compressing a downsampled OOI DAS data segments:
- Random Fourier Feature Network
    - training: [notebooks/rffn_training_ooi_10min_pytorch.ipynb](./notebooks/INR/rffn_training_ooi_10min.ipynb)
    - reconstruct: [notebooks/rffn_reconstruct_ooi_10min_pytorch.ipynb](./notebooks/INR/rffn_reconstruct_ooi_10mim.ipynb)
- SIREN

## Requirements
All codes are implemented in PyTorch (https://pytorch.org), but the TensorFlow implementation of some models are provided (see [notebooks](./notebooks/)). A NVidia A100 GPU is used for all tests. Note that codes are not tested on MacBook Pro with M1 chips.

## Data
Please see [data/README.md](./data/README.md) for instructions on downloading Cook Inlet DAS data.

## Reference
- Williams, J. P., Zahn, O., & Kutz, J. N. (2023). Sensing with shallow recurrent decoder networks. arXiv preprint arXiv:2301.12011.
- Sitzmann, V., Martel, J., Bergman, A., Lindell, D., & Wetzstein, G. (2020). Implicit neural representations with periodic activation functions. Advances in Neural Information Processing Systems, 33, 7462-7473
- Tancik, M., Srinivasan, P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., ... & Ng, R. (2020). Fourier features let networks learn high frequency functions in low dimensional domains. Advances in Neural Information Processing Systems, 33, 7537-7547.