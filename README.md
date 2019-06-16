# ML_cifar10

## PCA

- Python version: based on Python 3.6.7
- Main dependencies: numpy, tensorflow, matplotlib, sklearn.
- Cifar 10 dataset location: ./data/cifar-10-batches-bin/*
- How to run: `python main.py`

## SVM

- Python version: based on Python 3.6
- Main dependencies: numpy, matplotlib, sklearn.
- How to run: `jupyter notebook` open `CIFAR File`

## LSTM

The code is implemented in python with PyTorch library

1. Install dependencies

```sh
pip install -r requirements.txt
```

Refer to https://pytorch.org/get-started/locally/ if PyTorch installation failed.

2. Training

```sh
python cifar10_lstm.py --do_train -e EPOCH_TO_TRAIN --model_dir=PATH_TO_SAVE_MODEL
```

3. Evaluation

```sh
python cifar10_lstm.py --do_test --ckpt_dir=PATH_TO_SAVED_MODEL
```

## CNN

To run this program is simple. First, make sure you have `numpy`, `tensorflow` and `keras` installed. In order to train the model, run `python classification.py --mode train`. In order to evaluate the model, run `python classification.py --mode test`.



## Contribution of This Project

`PCA` : Rui Zhang

`SVM` : Chaoyi Wang

`LSTM` : Junnan Liu

`CNN` : Tianling Bian

