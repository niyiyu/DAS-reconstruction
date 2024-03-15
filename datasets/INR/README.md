# Dataset
There are two sets of files under this folder that are compressed DAS data. The training data could be downloaded at https://drive.google.com/file/d/1mtvCeygfLWp5jXkg_FZYezlcB4nbKM75/view?usp=share_link (275 MB) 
1. **test_OOI_10min_rffn_ 0.74 MB**

    These two files come from the test implemented in the [notebook](../notebooks/rffn_training_ooi_10min_pytorch.ipynb). The Random Fourier Feature Network is trained on a downsampled 10 minutes OOI DAS chunk (matrix original size 6000@10Hz X 6000 channel). The random encoding matrix (_feature.npy) and the model weights (_weights.pt) are freezed here. These two files contains all compressed inforamtion that are required for reconstruction (see [notebook](../notebooks/rffn_reconstruct_ooi_10min_pytorch.ipynb)).

