# UGATIT-pytorch-colab
Samples for training UGATIT in Google Colab with pytorch

[About UGATIT](https://arxiv.org/pdf/1907.10830.pdf)

## OVERVIEW
- [colab/](colab/) - Sample ipynb files for training or mobile model conversion
  - [train_example](colab/train_example.ipynb) - The template of training UGATIT model
- [models/](models/) - UGATIT networks (used in colab files)

## USAGE
### Training
1. Copy [train_example.ipynb](colab/train_example.ipynb) to your Google Drive storage.
2. Prepare datasets and put it in storage too.
3. Open train_example.ipynb and fill in necessary infomations (ex. the path to datasets in your drive).
4. Rewrite others (ex. hyper parameters) if you need.
5. Execute codes in order and start training (trained models are saved in drive).

### Mobile model conversion (pytorch-mobile/coreML)
There are two examples for [pytorch-mobile](colab/mobile/pytorch-mobile/pytorch-to-mobile.ipynb)(android) and [coreML](colab/mobile/coreml/pytorch-to-coreml.ipynb)(ios).

1. Copy the sample file for mobile model conversion to your Google Drive storage.
2. Open the file and fill in necessary infomations (ex. the path to trained models in your drive).
3. Rewrite others if necessary and execute codes in order.

### Quantization (only coreML)
1. Copy [the sample file](colab/mobile/coreml/coreml_quantization.ipynb) for quantization to your Google Drive storage.
2. Open the file and fill in necessary infomations (ex. the path to the mobile model in your drive).
3. Rewrite others if necessary and execute codes in order.

## NOTICE
Pytorch objects in /models is modified from https://github.com/znxlwm/UGATIT-pytorch/blob/master/networks.py
