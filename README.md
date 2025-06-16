# CCB

This is the code for:
**CCB: Enhancing Semi-Supervised Medical Image Segmentation with Perturbation Strategy, Label Adaptation and Boundary Optimization**</br>

## Getting Started

### Dataset download
- ISIC 2018: 
  - [office](https://challenge.isic-archive.com/data/#2018)
  - [ours](https://pan.baidu.com/s/1PKf0q3UzqXKMYkFCKvrwNw?pwd=ykve)
- Kvasir-SEG: 
  - [office](https://datasets.simula.no/kvasir-seg/)
  - [ours](https://pan.baidu.com/s/15zBUQTHb4tLsWWj8O9yEow?pwd=ih2b)
- ACDC: [office](https://drive.google.com/file/d/1LOust-JfKTDsFnvaidFOcAVhkZrgW1Ac/view?usp=sharing)

you can download datasets and put into your dataset path:
```
├── [Your Dataset Path]
    ├── ISIC 2018
    ├── Kvasir-SEG
    └── ACDC
```
label.txt/unlabel.txt saved at``splits``：
```
├── splits
    ├── ISIC2018
        ├──1_4
          ├──labeled.txt
          ├──unlabeled.txt
        ├──1_8
          ├──labeled.txt
          ├──unlabeled.txt
        ├──1_16
          ├──labeled.txt
          ├──unlabeled.txt
        ├──train.txt
        ├──val.txt
    ├── Kvasir-SEG
        ├──1_4
          ├──labeled.txt
          ├──unlabeled.txt
        ├──1_8
          ├──labeled.txt
          ├──unlabeled.txt
        ├──1_16
          ├──labeled.txt
          ├──unlabeled.txt
        ├──train.txt
        ├──val.txt
    └── ACDC
        ├──3
          ├──labeled.txt
          ├──unlabeled.txt
        ├──7
          ├──labeled.txt
          ├──unlabeled.txt
        ├──test.txt
        ├──val.txt
        ├──valtest.txt
        ├──train_slices.txt
```

### Train the model
To facilitate the configuration of hyperparameters and other settings, we uniformly use ``yaml``files to manage training parameters, which are all stored in the ``configs`` directory.

```
├── configs
    ├── isic2018.yaml
    ├── KvasirSEG.yaml
    └── ACDC.yaml
```

You need to modify the ``root_path`` in the yaml file to specify your dataset storage location.

If you want train model on ISIC 2018 or Kvasir-SEG, you can do:
```
python train_2d_CCB.py
```

If you want train model on ACDC, you can do:
```
python train_2d_ACDC_CCB.py
```

## Acknowledgement

This code is adapted from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) and [UniMatch](https://github.com/LiheYoung/UniMatch).
We thank Xiangde Luo and Lihe Yang for their elegant and efficient code base.

