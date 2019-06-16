# HOW TO USE
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

