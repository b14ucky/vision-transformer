# Vision Transformer

This project is my PyTorch implementation of a Vision Transformer (ViT) model, proposed in a research paper "[An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale](https://arxiv.org/abs/2010.11929)".\
ViTs divide images into a sequence of patches and apply transformer architectures---originally designed for text---to image recognition.

## Table of Contents
- [Vision Transformer](#vision-transformer)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Model Architecture](#model-architecture)
  - [Variations](#variations)
  - [Training](#training)
  - [Performance](#performance)
  - [Future Improvements](#future-improvements)
  - [Usage](#usage)
    - [Locally](#locally)
    - [Kaggle](#kaggle)
  - [Dataset](#dataset)
  - [Contributing](#contributing)
  - [License](#license)

## Project Overview

The goal of this project was to implement a Vision Transformer model for classifying images of fruits and vegetables. IIt was built step by step following [the original research paper](https://arxiv.org/abs/2010.11929). Working on this project helped me understand the transformer architecture and taught me a lot about training large models on multiple devices in parallel.

## Model Architecture

Transformers take a sequence as an input so to feed it an image it is divided into patches, to achieve this I used `nn.Unfold` class. Below is an example of an image divided into patches:
![patch division visualisation](https://github.com/b14ucky/vision-transformer/tree/main/images/patch-div-vis.png)
Each patch is then embedded using linear projection. Next a class token is prepended to the sequence, its purpose is to gather information about the image. Positional embeddings are added to both the patches and the class token to provide information about their original position in the image. The sequence is then passed through transformer encoders, and lastly the class token is processed through an MLP to obtain the final output.

## Variations

Due to the small dataset size (~85k samples in the training split and ~25k for validation) and limited computational resources, I created custom model variations. Initially, I attempted to train the smallest model from the paper---ViT-B---but after 2 epochs with no decrease in loss, I decided to develop my own models. The parameters for these custom models are shown below:
|    Model     | Layers | Embedding size | MLP size | Heads |
| :----------: | :----: | :------------: | :------: | :---: |
|   ViT-Tiny   |   12   |      192       |   768    |   4   |
|  ViT-Small   |   12   |      384       |   1536   |   6   |
| ViT-Small-12 |   12   |      384       |   1536   |  12   |
The pre-trained models can be found at [Kaggle](https://www.kaggle.com/models/b14ucky/vision-transformer).

## Training

The models were trained using 2x T4 GPUs available on **Kaggle**. To fully utilize the GPUs, I employed PyTorch's `nn.DataParallel` for multi-GPU training (`torch.nn.parallel.DistributedDataParallel` won't work as it uses multiprocessing wheras Jupyter is a process itself). Each model was trained for `10` epochs with a `learning rate` of `8e-5`. AdamW was used as the optimizer with a weight decay of `0.1`. The loss was evaluated using the `CrossEntropyLoss` function.

## Performance

After training, I evaluated all three models on the test dataset to compare their accuracy. The results are presented in the table below:
|  Model   |  Loss   | Accuracy |
| :------: | :-----: | :------: |
|  ViT-T   | 6.13117 |  38.68%  |
|  ViT-S   | 7.09262 |  41.46%  |
| ViT-S-12 | 7.22002 |  41.21%  |
The results indicate that all three models are overfitting. This is likely due to the dataset's relatively small size, as transformers generally perform better with larger datasets. Despite this, I am quite satisfied with the results. The ViT-S model was able to correctly classify photos of several fruits and vegetables that I found in my kitchen.

## Future Improvements

1. I want to investigate why the loss increases even as accuracy improves.
2. Adding more image augmentation techniques may help improve generalization and prevent the models from overfitting.
3. I also plan to add functionality to visualize attention maps, which would help me understand how the model processes different parts of the image---completely new samples that were not part of the test dataset. This shows some degree of robustness in real-world scenarios.

## Usage

### Locally
1. Clone the repository and install required dependencies:
```bash
	git clone git@github.com:b14ucky/vision-transformer.git
	cd vision-transformer
	pip install -r requirements.txt
```
2. The dataset is available [here](https://www.kaggle.com/datasets/sergeynesteruk/packed-fruits-and-vegetables-recognition-benchmark), and pre-trained models can be found [here](https://www.kaggle.com/models/b14ucky/vision-transformer).
3. Open the notebook in Jupyter, Kaggle, or your preferred environment.
4. Make sure to update the paths to your dataset and/or models locations.

### Kaggle
1. The notebook with models and dataset is also available and can be run [here](https://www.kaggle.com/code/b14ucky/vision-transformer).
2. Everything is already set up, so you can simply click the `Copy & Edit` button and run the notebook.

## Dataset

The dataset used for training the model - "[Packed Fruits and Vegetables Recognition Benchmark](https://www.kaggle.com/datasets/sergeynesteruk/packed-fruits-and-vegetables-recognition-benchmark)", contains images of 34 species and 65 varieties of fruits and vegetables.

## Contributing

Pull requests are welcome. Feel free to submit one if you would like to contribute.

## License

[MIT](https://choosealicense.com/licenses/mit/)