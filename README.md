# One-of-Many Learning

Working directory for [Learning One-of-Many Solutions for Combinatorial Problems in Structured Output Spaces](https://openreview.net/forum?id=ATp1nW2FuZL).
Use this repo for replicating experiments in paper exactly. For cleaner version of code and documentation please refer [https://github.com/dair-iitd/1oML](https://github.com/dair-iitd/1oML).

We experiment with 2 base models: [Neural Logic Machines (NLM)](https://arxiv.org/abs/1904.11694) and [Recurrent Relational Networks (RRN)](https://arxiv.org/abs/1711.08028).

Our training code has been adapted from [google/neural-logic-machines](https://github.com/google/neural-logic-machines) and uses [Jacinle python toolbox](https://github.com/vacancy/Jacinle).


## Installation
Before installation you need 
* python 3
* numpy
* tqdm
* yaml
* pandas
* PyTorch >= 1.5.0 (should probably work with >=1.0 versions, tested extensively with >=1.5)


Clone this repository:

```
git clone https://github.com/dair-iitd/1oML --recursive
```

Install [Jacinle](https://github.com/vacancy/Jacinle) included in `third_party/Jacinle`. You need to add the bin path to your global `PATH` environment variable:

```
export PATH=<path_to_1oML>/third_party/Jacinle/bin:$PATH
```

Create a conda environment for 1oML, and install the requirements. This includes the required python packages
from both Jacinle and NLM. Most of the required packages have been included in the built-in `anaconda` package:

```
conda create -n nlm anaconda
conda install pytorch torchvision -c pytorch
```


Install [dgl](https://github.com/dmlc/dgl) for RRN. 

## Replication

* Download datasets from this [drive link](https://drive.google.com/drive/folders/1s9QZXJGeAzCRuVcS7bSo7vQooCDRcTZe?usp=sharing). 
* Run experiments from `scripts/shell_commands/[nqueens|futo|sudoku]_e2e_[baselines|ccloss|minloss|Iexplr|selectR].sh`

