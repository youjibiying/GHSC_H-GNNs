# A Unified Random Walk, Its Induced  Laplacians and  Spectral Convolutions for Deep Hypergraph Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

The official implementation of the Journal paper [A Unified Random Walk, Its Induced  Laplacians and  Spectral Convolutions for Deep Hypergraph Learning](https://youjibiying.github.io/files/TPAMI-Under_review_A_Unified_Random_Walk.pdf).
Jiying Zhang, Fuyang Li, Xi Xiao, Guanzi Chen, Yu Li, Tingyang Xu, Yu Rong, Junzhou Huang, Yatao Bian

## Introduction

<!-- 
![](figures/.png) -->

## Getting Started

### Dependency

To run our code, the following Python libraries are required to run our code:

```
pytorch 1.8.0+ (torch1.12)
torch-geometric >2.0.0
torch-scatter
torch-sparse
torch-cluster
```
The complete conda env can be found in [torch1.12.yaml](./torch1.12.yaml)

### Data Preparation for ED-HNN split

Download the preprocessed dataset from the [HuggingFace Hub](https://huggingface.co/datasets/peihaowang/edgnn-hypergraph-dataset).
Then put the downloaded directory under the root folder of this repository. The directory structure should look like:
```
GHSC_H-GNNS/data
  ...
  raw_data
    coauthorship
    cocitation
    ...
```

### Data Preparation for HyperGCN split （The table in the main paper）
1. citation network:
directly use the data split from the [HuggingFace Hub](https://huggingface.co/datasets/peihaowang/edgnn-hypergraph-dataset) above
or download it from https://github.com/youjibiying/H-GNNs/tree/main/data
2. Visual object classification
Following [HGNN](http://gaoyue.org/paper/HGNN.pdf), the datasets can download as below for training/evaluation 
- [ModelNet40_mvcnn_gvcnn_feature](https://drive.google.com/file/d/1euw3bygLzRQm_dYj1FoRduXvsRRUG2Gr/view?usp=sharing)
- [NTU2012_mvcnn_gvcnn_feature](https://drive.google.com/file/d/1Vx4K15bW3__JPRV0KUoDWtQX8sB-vbO5/view?usp=sharing)

Then put it into the raw_data
```
data/raw_data/NTU2012_mvcnn_gvcnn.mat
data/raw_data/ModelNet40_mvcnn_gvcnn.mat
```
## Train and test



###   H-GCNII for HyperGCN split (--no_random_split)

- TABLE 1 in the main paper
```
python train.py --method H_GCNII --dname cora --lr 0.001 --degree 32 --MLP_hidden 128 --wd 0.001 --epochs 500 --runs 10 --cuda 0 --data_dir data/cocitation/cora --raw_data_dir data/raw_data/cocitation/cora --no_random_split
python train.py --method H_GCNII --dname citeseer --lr 0.001 --degree 2 --MLP_hidden 128 --wd 0.001 --epochs 500 --runs 10 --cuda 0 --data_dir data/cocitation/citeseer --raw_data_dir data/raw_data/cocitation/citeseer --no_random_split
python train.py --method H_GCNII --dname pubmed --lr 0.001 --degree 4 --MLP_hidden 512 --wd 0.001 --epochs 600 --runs 10 --cuda 3 --data_dir data/cocitation/pubmed --raw_data_dir data/raw_data/cocitation/pubmed --no_random_split
python train.py --method H_GCNII --dname coauthor_dblp --lr 0.001 --degree 32 --MLP_hidden 128 --wd 0.001 --epochs 500 --runs 10 --cuda 1 --data_dir data/coauthorship/dblp --raw_data_dir data/raw_data/coauthorship/dblp --no_random_split
python train.py --method H_GCNII --dname coauthor_cora --lr 0.001 --degree 32 --MLP_hidden 128 --wd 0.001 --epochs 500 --runs 10 --cuda 2 --data_dir data/coauthorship/cora --raw_data_dir data/raw_data/coauthorship/cora --no_random_split
```

- TABLE 2 in the main paper
```
python train.py --method H_GCNII --dname NTU2012_large --lr 0.001 --degree 2 --MLP_hidden 128 --wd 0.005 --epochs 500 --runs 10 --cuda 1 --data_dir data/NTU2012_large --raw_data_dir data/raw_data/ --no_random_split --no_mvcnn_feature_structure --no_use_mvcnn_feature &
python train.py --method H_GCNII --dname NTU2012_large --lr 0.001 --degree 4 --MLP_hidden 256 --wd 0.005 --epochs 500 --runs 10 --cuda 2 --data_dir data/NTU2012_large --raw_data_dir data/raw_data/ --no_random_split --no_gvcnn_feature_structure --no_use_gvcnn_feature
python train.py --method H_GCNII --dname NTU2012_large --lr 0.001 --degree 4 --MLP_hidden 256 --wd 0.005 --epochs 500 --runs 10 --cuda 0 --data_dir data/NTU2012_large --raw_data_dir data/raw_data/ --no_random_split
python train.py --method H_GCNII --dname ModelNet40_large --lr 0.001 --degree 2 --MLP_hidden 128 --wd 0.005 --epochs 500 --runs 10 --cuda 3  --data_dir data/ModelNet40_large --raw_data_dir data/raw_data/ --no_random_split --no_gvcnn_feature_structure --no_use_gvcnn_feature --H_GNN_alpha 0.05 --H_GNN_lamda 0.5 --dropout 0.2 
python train.py --method H_GCNII --dname ModelNet40_large --lr 0.001 --degree 2 --MLP_hidden 256 --wd 0.005 --epochs 500 --runs 10 --cuda 0 --data_dir data/ModelNet40_large --raw_data_dir data/raw_data/ --no_random_split --no_mvcnn_feature_structure --no_use_mvcnn_feature
python train.py --method H_GCNII --dname ModelNet40_large --lr 0.001 --degree 2 --MLP_hidden 128 --wd 0.005 --epochs 500 --runs 10 --cuda 0 --data_dir data_1/data/ModelNet40_large --raw_data_dir data_1/data/raw_data/ --no_random_split --H_GNN_alpha 0.3 --H_GNN_lamda 0.55 --dropout 0.5
 
 ```
 
### H-GCNII for ED-HNN split

- TABLE 1 in the summary of changes
```
python train.py --method H_GCNII --dname cora --lr 0.001 --degree 32 --MLP_hidden 128 --wd 0.001 --epochs 500 --runs 10 --cuda 0 --data_dir data/cocitation/cora --raw_data_dir data/raw_data/cocitation/cora
python train.py --method H_GCNII --dname citeseer --lr 0.001 --degree 2 --MLP_hidden 128 --wd 0.001 --epochs 500 --runs 10 --cuda 0 --data_dir data/cocitation/citeseer --raw_data_dir data/raw_data/cocitation/citeseer
python train.py --method H_GCNII --dname coauthor_cora --lr 0.001 --degree 32 --MLP_hidden 128 --wd 0.001 --epochs 500 --runs 10 --cuda 2 --data_dir data/coauthorship/cora --raw_data_dir data/raw_data/coauthorship/cora
python train.py --method H_GCNII --dname pubmed --lr 0.001 --degree 4 --MLP_hidden 256 --wd 0.0001 --epochs 500 --runs 10 --cuda 3 --data_dir data/cocitation/pubmed --raw_data_dir data/raw_data/cocitation/pubmed --H_GNN_alpha 0.1 --H_GNN_lamda 0.6 --dropout 0.5 &
python train.py --method H_GCNII --dname coauthor_dblp --lr 0.001 --degree 4 --MLP_hidden 128 --wd 0.001 --epochs 500 --runs 10 --cuda 1 --data_dir data/coauthorship/dblp --raw_data_dir data/raw_data/coauthorship/dblp --H_GNN_alpha 0.1 --H_GNN_lamda 0.6 --dropout 0.5 &

python train.py --method H_GCNII --dname senate-committees-100 --lr 0.001 --degree 32 --MLP_hidden 128 --wd 0.001 --epochs 500 --runs 10 --cuda 1 --data_dir data/senate-committees --raw_data_dir data/raw_data/senate-committees --feature_noise 1
python train.py --method H_GCNII --dname house-committees-100 --lr 0.001 --degree 32 --MLP_hidden 128 --wd 0.001 --epochs 500 --runs 10 --cuda 1 --data_dir data/house-committees --raw_data_dir data/raw_data/house-committees --feature_noise 1
```
 



## Citation

If you find this work or our code implementation helpful for your research or work, please cite our paper.
```
@inproceedings{zhang2025unified,
  title={ A Unified Random Walk, Its Induced  Laplacians and  Spectral Convolutions for Deep Hypergraph Learning},
  author={ Jiying Zhang, Fuyang Li, Xi Xiao, Guanzi Chen, Yu Li, Tingyang Xu, Yu Rong, Junzhou Huang, Yatao Bian},
  booktitle={Arxiv},
  year={2025}
}
```
We would like to appreciate the excellent work of ED-HNN ([official repository](https://github.com/Graph-COM/ED-HNN)), which lays a solid foundation for our work.

