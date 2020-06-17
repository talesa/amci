# Amortized Monte Carlo Integration

This repo provides the code accompanying two papers:  
A. Goliński*, F. Wood, T. Rainforth*, [*Amortized Monte Carlo Integration*](http://proceedings.mlr.press/v97/golinski19a.html), ICML, 2019  
T. Rainforth*, A. Goliński*, F. Wood, S. Zaidi, [*Target–Aware Bayesian Inference: How to Beat Optimal Conventional Estimators*](http://jmlr.org/papers/v21/19-102.html), 
JMLR, 2020

## Requirements
The default config files included in this code (particularly the data set and batch size settings) 
requires a GPU with ~8GB RAM for training the neural networks and generating samples required to reproduce the figures from the papers.
The samples generated are stored onto the hard drive what requires about 5.1GB of disk space in total for all the experiments.

## Environment setup
Make sure to set the appropriate version of `cudatoolkit` for your environment below
```
conda create -n amci python=3.7
conda activate amci
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=X.X -c pytorch
conda install --file requirements_conda.txt
pip install -r requirements_pip.txt
```

## Instructions 

0. `cd` into the repo, then `cd src; export PYTHONPATH=$(pwd); cd amci`
0. Train `q1` and `q2` networks by running 
    ```
    python train.py with tail_integral/config_1d.yaml q1_or_q2=q1 --force 
    python train.py with tail_integral/config_1d.yaml q1_or_q2=q2 --force
    python train.py with tail_integral/config_5d.yaml q1_or_q2=q1 --force 
    python train.py with tail_integral/config_5d.yaml q1_or_q2=q2 --force
    python train.py with cancer/config.yaml q1_or_q2=q1 factor=c0 --force
    python train.py with cancer/config.yaml q1_or_q2=q1 factor=eps --force
    python train.py with cancer/config.yaml q1_or_q2=q2 factor=c0 --force
    python train.py with cancer/config.yaml q1_or_q2=q2 factor=eps --force
    ``` 
    After the training is finished update `checkpoint_q1` and `checkpoint_q2` fields in the respective `config*.yaml` 
     files with the paths of the checkpoints generated in `logs` directory.
    For the `cancer` example you also have to update the `checkpoint_q1/q2_eps/c0` fields (aside from `checkpoint_q1` and `checkpoint_q2`). 
0. To generate samples from the learned proposals that will be combined into estimates in the next step run
    ```
    python generate_samples.py tail_integral/config_1d.yaml
    python generate_samples.py tail_integral/config_5d.yaml
    python generate_samples.py cancer/config.yaml 
    ```
    The script saves the generated samples onto the hard drive and hence consumes a considerable amount of data, about 5.1GB in total for all the experiments. 
0. To reproduce the ReMSE figures from the paper run 
    ```
    python create_remse_figure_from_samples.py tail_integral/config_1d.yaml
    python create_remse_figure_from_samples.py tail_integral/config_5d.yaml
    python create_remse_figure_from_samples.py cancer/config.yaml
    ```
    The pdf figures are saved in the respective checkpoint folders.

#### Ground truth
The ground truth estimates included for the `tail_integral_5d` and `cancer` experiments were generated using `1e10` importance samples with prior as the proposal.
If you wish to regenerate them run
```
python ground_truth.py tail_integral/config_5d.yaml
python ground_truth.py cancer/config.yaml 
```
For the `tail_integral_1d` there is no need to estimate the ground truth because it is obtained analytically.

## Miscellaneous
Many thanks to the developers and contributors of [github.com/ikostrikov/pytorch-flows](https://github.com/ikostrikov/pytorch-flows/) and [github.com/facebookresearch/higher](https://github.com/facebookresearch/higher/).

If anything is unclear, doesn't work, or you just have questions please create an issue, I'll try to get back to you.

## Cite this work
```
@inproceedings{golinski2019amortized,
  title     = {{A}mortized {M}onte {C}arlo {I}ntegration},
  author    = {Goli{\'n}ski, Adam and Wood, Frank and Rainforth, Tom},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2019},
}

@article{rainforth2020target,
  author  = {Tom Rainforth and Adam Goli{\'n}ski and Frank Wood and Sheheryar Zaidi},
  title   = {{T}arget--{A}ware {B}ayesian {I}nference: {H}ow to Beat Optimal Conventional Estimators},
  journal = {Journal of Machine Learning Research (JMLR)},
  year    = {2020},
  volume  = {21},
  number  = {88},
  pages   = {1-54},
}
```
