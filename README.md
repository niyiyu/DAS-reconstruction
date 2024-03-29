# Distributed Acoustic Sensing Compression and Reconstruction with Machine Learning
![](./figures/reconstruction.png)

## Tutorials
All codes are implemented in PyTorch.
![SHRED](./figures/shred.png)
- SHallow REcurrent Decoder (SHRED): [notebooks/SHRED_KKFLS_training.ipynb](./notebooks/SHRED_KKFLS_training.ipynb)

#### Implicit Neural Representation (INR)
![SIREN_vs_RFFN](./figures/siren_vs_rffn_50_40epoch.gif)
- Random Fourier Feature Network (RFFN): [notebooks/RFFN_KKFLS_training.ipynb](./notebooks/RFFN_KKFLS_training.ipynb)
- Sinusoidal Representation Network (SIREN): [notebooks/SIREN_KKFLS_training.ipynb](./notebooks/SIREN_KKFLS_training.ipynb)

## Data
Please see [data/README.md](./data/README.md) for instructions on downloading Cook Inlet DAS data.

## Reference
- Williams, J. P., Zahn, O., & Kutz, J. N. (2023). Sensing with shallow recurrent decoder networks. arXiv preprint arXiv:2301.12011.
- Sitzmann, V., Martel, J., Bergman, A., Lindell, D., & Wetzstein, G. (2020). Implicit neural representations with periodic activation functions. Advances in Neural Information Processing Systems, 33, 7462-7473
- Tancik, M., Srinivasan, P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., ... & Ng, R. (2020). Fourier features let networks learn high frequency functions in low dimensional domains. Advances in Neural Information Processing Systems, 33, 7537-7547.