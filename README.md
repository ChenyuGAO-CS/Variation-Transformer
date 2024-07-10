# Variation-Transformer
This repository contains the source code to reproduce the results of the paper: Gao et al., Variation Transformer: New datasets, models, and comparative evaluation for symbolic music variation generation, in _ISMIR_ 2024.

## Demo page
https://variation-transformer.glitch.me

All the materials used in our listening study have been uploaded to our demo page too. 

## Trained models and datasets
https://github.com/ChenyuGAO-CS/Variation-Transformer-Data-and-Model

## Dependencies
```
pip install requirements.txt
```

## Reproducing Results
To generate variations using models trained by us, please visit [this page](https://github.com/ChenyuGAO-CS/Variation-Transformer-Data-and-Model) to download corresponding models and datasets. 

We will show examples of how to use a model trained on the POP909-TVar dataset to generate a variation below. Please change ```--lm``` and ```--input``` if you would like to try models trained on the VGMIDI-TVar dataset or other themes as input. 

### 1. Generate a variation by using the Variation Transformer
Run the script ```gen_VaTr_var_user_input.py``` from the ```workspace``` folder. 

```
$ python3 workspace/gen_VaTr_var_user_input.py --lm trained_models/VaTr_pop909_epoch_10.pth \ 
                           --seq_len 1025 \ 
                           --n_bars 16 \
                           --p 0.9 \
                           --save_to out/VaTr_variation1.mid \
                           --input dataset/POP909-TVar/test/052_B_0.mid
```

### 2. Generate a variation by using the Music Transformer
Run the script ```gen_MuTr_var_user_input.py``` from the ```workspace``` folder. 

```
$ python3 workspace/gen_MuTr_var_user_input.py --lm trained_models/MuTr_pop909_epoch_10.pth \ 
                           --seq_len 1025 \ 
                           --n_bars 16 \
                           --p 0.9 \
                           --save_to out/MuTr_variation1.mid \
                           --input dataset/POP909-TVar/test/052_B_0.mid
```

### 3. Generate a variation by using the fast-Transformer
Run the script ```gen_FaTr_var_user_input.py``` from the ```workspace``` folder. 

```
$ python3 workspace/gen_FaTr_var_user_input.py --lm trained_models/FaTr_pop909_epoch_10.pth \ 
                           --seq_len 1025 \ 
                           --n_bars 16 \
                           --p 0.9 \
                           --save_to out/FaTr_variation1.mid \
                           --input dataset/POP909-TVar/test/052_B_0.mid
```
