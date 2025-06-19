# Code for LIFT

![Paper](https://img.shields.io/badge/Paper-ISWC%202025-blue)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10%2B-red)](https://pytorch.org/)

This repository contains the official implementation of ​**LIFT**.

## Requirements
The code has been tested running under Python 3.8.0. Required packages:
- numpy == 1.22.4
- pandas == 1.4.3
- scikit-learn == 1.1.1
- scipy == 1.7.0
- networkx == 2.5.1
- tqdm == 4.64.1  
- torch == 1.10.1+cu113
- torch-cluster == 1.5.9+pt110cu113
- torch-scatter == 2.0.9+pt110cu113
- torch-sparse == 0.6.12+pt110cu113
- torch-geometric == 1.7.2 

## Dataset

We provide three processed datasets: Yelp, Tmall, and Retail.

We follow the paper " [Knowledge-Enhanced Hierarchical Graph Transformer Network
for Multi-Behavior Recommendation](https://github.com/akaxlh/KHGT)." to process data.


| Dataset | Users   | Items  | Interactions | Behavior Type                  |
| ------- | ------- | ------ | ------------ | ------------------------------ |
| Yelp    | 19,800  | 22,734 | 1,400,002    | Dislike, Neutral, Tip, Like    |
| Tmall   | 31,882  | 31,232 | 1,451,219    | View, Favorite, Cart, Purchase |
| Retail  | 147,894 | 99,037 | 1,584,238    | Favorite, Cart, Purchase       | 


The other two datasets are available from (https://github.com/akaxlh/KHGT).

## Reference 
- We partially use the codes of [KMCLR].


## Usage
To run the LIFT model on different datasets, execute the main script with the corresponding dataset parameter:

```bash
python main.py --dataset Yelp

python main.py --dataset Tmall

python main.py --dataset retail
