# Pretrained SRSdenoiser Neural Networks

This directory contains the optimized weights and network models of the three SRSdenoiser best networks discussed in the paper. 
* `NN_HN` is the model trained on the high noise dataset, which is discussed in the main text of the paper.
* `NN_HN_reduced` is the version of the model trained on the high noise dataset having a reduced number of parameters, which is discussed in the main text of the paper.
* `NN_LN` is the model trained on the low noise dataset, which is discussed in the main text of the paper.


The hyperparameter settings of the three models is summarized below:


| Hyperparameter   |      NN_HN      | NN_HN_reduced| NN_LN |
| -------------    |:-------------:| :-----:|:------: |
| N<sub>conv</sub> |0          |0           |4        |
| N<sub>param</sub>| 11        |   10       | 10 |
| N<sub>kernel</sub>  | 63     |   21       | 21  |
| N<sub>batch size</sub>  | 32      |    32 |  32 |
| N<sub>epoch</sub><sup>0</sup>    | 25      |    25 | 25 |
| N<sub>epoch</sub><sup>1</sup>  | 200|   200 | 200  |
| *W*<sub>grad</sub>  | 0.6|   0.6 | 0.6  |
| *l*| 2|   1 | 2  |