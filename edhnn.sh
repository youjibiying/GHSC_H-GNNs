## HyperGCN data split
python train.py --method EDGNN --dname coauthor_cora --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 --MLP3_num_layers 1 --Classifier_num_layers 2 --MLP_hidden 128 --Classifier_hidden 96 --aggregate mean --restart_alpha 0.0 --lr 0.001 --wd 0 --epochs 500 --runs 10 --cuda 1 --data_dir data/coauthorship/cora --raw_data_dir data/raw_data/coauthorship/cora --no_random_split
python train.py --method EDGNN --dname coauthor_dblp --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 --MLP3_num_layers 1 --Classifier_num_layers 2 --MLP_hidden 128 --Classifier_hidden 96 --aggregate mean --restart_alpha 0.0 --lr 0.001 --wd 0 --epochs 500 --runs 10 --cuda 0 --data_dir data/coauthorship/dblp --raw_data_dir data/raw_data/coauthorship/dblp --no_random_split
python train.py --method EDGNN --dname cora --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 --MLP3_num_layers 1 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 256 --aggregate mean --restart_alpha 0.0 --lr 0.001 --wd 0 --epochs 500 --runs 10 --cuda 0 --data_dir data/cocitation/cora --raw_data_dir data/raw_data/cocitation/cora --no_random_split
python train.py --method EDGNN --dname citeseer --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 --MLP3_num_layers 1 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 256 --aggregate mean --restart_alpha 0.0 --lr 0.001 --wd 0 --epochs 500 --runs 10 --cuda 1 --data_dir data/cocitation/citeseer --raw_data_dir data/raw_data/cocitation/citeseer --no_random_split
python train.py --method EDGNN --dname pubmed --All_num_layers 8 --MLP_num_layers 2 --MLP2_num_layers 2 --MLP3_num_layers 2 --Classifier_num_layers 2 --MLP_hidden 512 --Classifier_hidden 256 --normalization None --aggregate mean --restart_alpha 0.5 --lr 0.001 --wd 0 --epochs 500 --runs 10 --cuda 0 --data_dir data/cocitation/pubmed --raw_data_dir data/raw_data/cocitation/pubmed --no_random_split

 python train.py --method EDGNN --dname NTU2012_large --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 --MLP3_num_layers 1 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 256 --aggregate mean --restart_alpha 0.0 --lr 0.001 --wd 0 --epochs 500 --runs 10 --cuda 0  --data_dir data/NTU2012_large --raw_data_dir data/raw_data/ --no_random_split --no_mvcnn_feature_structure --no_use_mvcnn_feature
 python train.py --method EDGNN --dname NTU2012_large --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 --MLP3_num_layers 1 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 256 --aggregate mean --restart_alpha 0.0 --lr 0.001 --wd 0 --epochs 500 --runs 10 --cuda 0  --data_dir data/NTU2012_large --raw_data_dir data/raw_data/ --no_random_split --no_gvcnn_feature_structure --no_use_gvcnn_feature
 python train.py --method EDGNN --dname NTU2012_large --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 --MLP3_num_layers 1 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 256 --aggregate mean --restart_alpha 0.0 --lr 0.001 --wd 0 --epochs 500 --runs 10 --cuda 0  --data_dir data/NTU2012_large --raw_data_dir data/raw_data/ --no_random_split
 
 python train.py --method EDGNN --dname ModelNet40_large --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 --MLP3_num_layers 1 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 256 --aggregate mean --restart_alpha 0.0 --lr 0.001 --wd 0 --epochs 500 --runs 10 --cuda 0  --data_dir data/ModelNet40_large --raw_data_dir data/raw_data/ --no_random_split --no_mvcnn_feature_structure --no_use_mvcnn_feature
 python train.py --method EDGNN --dname ModelNet40_large --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 --MLP3_num_layers 1 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 256 --aggregate mean --restart_alpha 0.0 --lr 0.001 --wd 0 --epochs 500 --runs 10 --cuda 1  --data_dir data/ModelNet40_large --raw_data_dir data/raw_data/ --no_random_split --no_gvcnn_feature_structure --no_use_gvcnn_feature
 python train.py --method EDGNN --dname ModelNet40_large --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 --MLP3_num_layers 1 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 256 --aggregate mean --restart_alpha 0.0 --lr 0.001 --wd 0 --epochs 500 --runs 10 --cuda 2  --data_dir data/ModelNet40_large --raw_data_dir data/raw_data/ --no_random_split
 
```


## ED-HNN data split
<summary>Cora</summary>

```
python train.py --method EDGNN --dname cora --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 --MLP3_num_layers 1 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 256 --aggregate mean --restart_alpha 0.0 --lr 0.001 --wd 0 --epochs 500 --runs 10 --cuda 0 --data_dir data/cocitation/cora --raw_data_dir data/raw_data/cocitation/cora --no_random_split

```

</details>

<details>

<summary>Citeseer</summary>

```
python train.py --method EDGNN --dname citeseer --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 --MLP3_num_layers 1 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 256 --aggregate mean --restart_alpha 0.0 --lr 0.001 --wd 0 --epochs 500 --runs 10 --cuda 1 --data_dir data/cocitation/citeseer --raw_data_dir data/raw_data/cocitation/citeseer --no_random_split 
```

</details>


<details>

<summary>Pubmed</summary>

```
python train.py --method EDGNN --dname pubmed --All_num_layers 8 --MLP_num_layers 2 --MLP2_num_layers 2 --MLP3_num_layers 2 --Classifier_num_layers 2 --MLP_hidden 512 --Classifier_hidden 256 --normalization None --aggregate mean --restart_alpha 0.5 --lr 0.001 --wd 0 --epochs 500 --runs 10 --cuda 0 --data_dir data/cocitation/pubmed --raw_data_dir data/raw_data/cocitation/pubmed

</details>


<details>

<summary>Cora-CA</summary>

```
python train.py --method EDGNN --dname coauthor_cora --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 --MLP3_num_layers 1 --Classifier_num_layers 2 --MLP_hidden 128 --Classifier_hidden 96 --aggregate mean --restart_alpha 0.0 --lr 0.001 --wd 0 --epochs 500 --runs 10 --cuda 0 --data_dir data/coauthorship/cora --raw_data_dir data/raw_data/coauthorship/cora 
```

</details>

<details>

<summary>DBLP-CA</summary>

```
python train.py --method EDGNN --dname coauthor_dblp --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 --MLP3_num_layers 1 --Classifier_num_layers 2 --MLP_hidden 128 --Classifier_hidden 96 --aggregate mean --restart_alpha 0.0 --lr 0.001 --wd 0 --epochs 500 --runs 10 --cuda 0 --data_dir data/coauthorship/dblp --raw_data_dir data/raw_data/coauthorship/dblp
```

</details>


<details>

<summary>Senate Committees</summary>

```
python train.py --method EDGNN --dname senate-committees-100 --All_num_layers 8 --MLP_num_layers 2 --MLP2_num_layers 2
--MLP3_num_layers 2 --Classifier_num_layers 2 --MLP_hidden 512 --Classifier_hidden 256 --aggregate mean 
--restart_alpha 0.5 --lr 0.001 --wd 0 --epochs 500 --runs 10 --feature_noise 1.0
--cuda <cuda_id> --data_dir <data_path> --raw_data_dir <raw_data_path>
```

</details>

<details>

<summary>House Committees</summary>

```
python train.py --method EDGNN --dname house-committees-100 --All_num_layers 8 --MLP_num_layers 2 --MLP2_num_layers 2
--MLP3_num_layers 2 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --aggregate mean 
--restart_alpha 0.5 --lr 0.001 --wd 0 --epochs 500 --runs 10 --feature_noise 1.0
--cuda <cuda_id> --data_dir <data_path> --raw_data_dir <raw_data_path>
```

</details>