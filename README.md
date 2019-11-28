# MolGAN
Tensorflow implementation of MolGAN: An implicit generative model for small molecular graphs (https://arxiv.org/abs/1805.11973)

## Overview
This library contains a Tensorflow implementation of MolGAN: An implicit generative model for small molecular graphs as presented in [[1]](#citation)(https://arxiv.org/abs/1805.11973).
## Dependencies

* **python>=3.6**
* **tensorflow>=1.7.0**: https://tensorflow.org
* **rdkit**: https://www.rdkit.org
* **numpy**
* **scikit-learn**

## Structure
* [data](https://github.com/nicola-decao/MolGAN/tree/master/data): should contain your datasets. If you run `download_dataset.sh` the script will download the dataset used for the paper (then you should run `utils/sparse_molecular_dataset.py` to convert the dataset in a graph format used by MolGAN models).
* [example](https://github.com/nicola-decao/MolGAN/blob/master/example.py): Example code for using the library within a Tensorflow project. **NOTE: these are NOT the experiments on the paper!**
* [models](https://github.com/nicola-decao/MolGAN/tree/master/models): Class for Models. Both VAE and (W)GAN are implemented.
* [optimizers](https://github.com/nicola-decao/MolGAN/tree/master/optimizers): Class for Optimizers for both VAE, (W)GAN and RL.

## Usage
Please have a look at the [example](https://github.com/nicola-decao/MolGAN/blob/master/example.py).

Please cite [[1](#citation)] in your work when using this library in your experiments.

## Feedback
For questions and comments, feel free to contact [Nicola De Cao](mailto:nicola.decao@gmail.com).

## License
MIT

## Citation
```
[1] De Cao, N., and Kipf, T. (2018).MolGAN: An implicit generative 
model for small molecular graphs. ICML 2018 workshop on Theoretical
Foundations and Applications of Deep Generative Models.
```

BibTeX format:
```
@article{de2018molgan,
  title={{MolGAN: An implicit generative model for small
  molecular graphs}},
  author={De Cao, Nicola and Kipf, Thomas},
  journal={ICML 2018 workshop on Theoretical Foundations 
  and Applications of Deep Generative Models},
  year={2018}
}

```
