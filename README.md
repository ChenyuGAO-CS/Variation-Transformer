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

## Model Training
### 1. Download the datasets
Please visit [this page](https://github.com/ChenyuGAO-CS/Variation-Transformer-Data-and-Model) to download POP909-TVar or VGMIDI-TVar. 

### 2. Data preprocessing for variation generation:

The pre-processing step consists of augmenting the data, encoding it with REMI and compiling the encoded pieces as a numpy array. Please find scripts for data pre-processing in the ```'dataset'``` folder.

Each theme-variation pair will be stored in a line of the numpy array, in which the token '520' will be used to separate the theme sequence and the variation sequence. 

**2.1. Data Augmentation**

All pieces were augmented by (a) transposing to every key, and (b) increasing and decreasing the tempo by 10% as Oore et al. (2017) and Ferreira et al. (2022) described.

```
$ python3 augment.py --path_indir POP909-TVar --path_outdir 909_augmented
```

**2.2. REMI Encoding**

We encoded all pieces using REMI (Huang and Yang, 2020).

```
$ python3 encoder.py --path_indir 909_augmented --path_outdir 909_encoded
```

**2.3. Compile pieces in a numpy array.**

Then, compile the theme-and-variation pairs for model training. The token '520' will be used to separate the theme sequence and the variation sequence. 

Each piece whose name starts with "songNum_phraseNum_0" will be used as a theme, which will be combined with all other pieces with the same song number, phrase number, key signature, and tempo. 

Run the script below to compile pieces from the POP909-TVar dataset in a numpy array:

```
$ python3 compile_for_var_gen_909.py --path_train_indir 909_encoded/train --path_test_indir 909_encoded/test --path_outdir 909_compiled --max_len 512 --task language_modeling
```

Run the script below to compile pieces from the VGMIDI-TVar dataset in a numpy array

```
$ python3 compile_for_var_gen_vgmidi.py --path_train_indir vgmidi_encoded/train --path_test_indir vgmidi_encoded/test --path_outdir vgmidi_compiled --max_len 512 --task language_modeling
```

## Theme-and-variation extraction
Please follow the steps on this [page](https://github.com/ChenyuGAO-CS/theme-variation-data-preprocessing) if you are interested in running our theme-and-variation extraction algorithms on your datasets.

## Citing this Work
If you use our method in your research, please cite:
```
@inproceedings{gao2024variation,
  title={Variation Transformer: New datasets, models, and comparative evaluation for symbolic music variation generation},
  author={Chenyu Gao, Federico Reuben, and Tom Collins},
  booktitle={the 25th International Society for Music Information Retrieval Conference},
  year={2024}
}
```
