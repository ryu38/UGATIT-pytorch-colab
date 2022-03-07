# UGATIT-pytorch-colab
The Examples for training UGATIT in Google Colab with using PyTorch

[About UGATIT](https://arxiv.org/pdf/1907.10830.pdf)

## OVERVIEW
- [colab/](colab/) - Sample ipynb files for training or mobile model conversion
  - [train_example.ipynb](colab/train_example.ipynb) - The template of training UGATIT models
- [models/](models/) - UGATIT networks (used in Colab files)

## USAGE
### Training
1. Copy [train_example.ipynb](colab/train_example.ipynb) to your Google Drive storage.
2. Prepare datasets and put them in the storage too.
3. Open train_example.ipynb and fill in necessary informations (ex. the path to datasets in your storage).
4. Modify others (ex. hyper parameters) if you need.
5. Execute blocks of code in order and start training (trained models are saved in the storage).

### Mobile model conversion (pytorch-mobile/coreML)
There are two examples for [PyTorch-Mobile](colab/mobile/pytorch-mobile/pytorch-to-mobile.ipynb)(android) and [coreML](colab/mobile/coreml/pytorch-to-coreml.ipynb)(ios).

1. Copy the sample file for mobile model conversion to your Google Drive storage.
2. Open the file and fill in necessary informations (ex. the path to trained models in your storage).
3. Modify others if necessary and execute blocks of code in order.

### Quantization (only coreML)
1. Copy [the sample file](colab/mobile/coreml/coreml_quantization.ipynb) for quantization to your Google Drive storage.
2. Open the file and fill in necessary informations (ex. the path to the mobile model in your storage).
3. Modify others if necessary and execute blocks of code in order.

## NOTICE
PyTorch objects in [models/](models/) has been modified from https://github.com/znxlwm/UGATIT-pytorch/blob/master/networks.py.
