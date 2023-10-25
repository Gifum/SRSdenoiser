# SRSdenoiser
This repository contains the implementation of the methods described in:

*Retrieving genuine nonlinear Raman responses in ultrafast spectroscopy via deep learning* [arXiv](https://arxiv.org/abs/2309.16933) | [PDF](https://arxiv.org/pdf/2309.16933.pdf) 

## Usage
The script `SRSdenois_main.ipynb` contains minimal implementation examples for loading the two datasets provided in `Datasets`, training the NN and testing the pretrained NNs provided in `Weights` on unseen examples from the datasets.

The script `metrics_2.py` in `Models` contains python implementations of the edge finder and the custom metrics described in the paper.

`Weights/weights_readme.md` and `Datasets/Datasets_readme` contain additional information on the pretrained networks and the datasets. Please refer to the main text and supplementary information of the article for further details.



## Cite
If you find this useful for your research, please consider citing:
```
@article{SRSdenoiser2023,
  doi = {10.48550/ARXIV.2309.16933},
  url = {https://arxiv.org/abs/2309.16933},
  author = {Fumero,  Giuseppe and Batignani,  Giovanni and Cassetta,  Edoardo and Ferrante,  Carino and Giagu,  Stefano and Scopigno,  Tullio},
  title = {Retrieving genuine nonlinear Raman responses in ultrafast spectroscopy via deep learning},
  publisher = {arXiv},
  year = {2023}
}
```
