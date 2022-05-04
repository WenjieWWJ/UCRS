# User-controllable Recommendation Against Filter Bubbles


This is the pytorch implementation of our paper at SIGIR 2022:

> [User-controllable Recommendation Against Filter Bubbles](https://arxiv.org/abs/2204.13844)
>
> Wenjie Wang, Fuli Feng, Liqiang Nie, Tat-Seng Chua.

## Environment

- Anaconda 3
- python 3.7.3
- pytorch 1.4.0
- numpy 1.16.4

## Usage

### Data

- The three datasets are released in the './data' folder.

### Code

- The code for training and inference is in the './code' folder. 
- FM and NFM are first well trained, and then UCI is used for inference. 
- We have user-feature controls (i.e., C-UCI and F-UCI), and item-feature controls (i.e., Reranking, C-UCI, and F-UCI).

#### FM and NFM Training
```
python main.py --model=$1 --dataset=$2 --hidden=$3 --layers=$4 --lr=$5 --batch_size=$6 --dropout=$7 --lamda=$8 --batch_norm=$9 --epochs=$10 --log_name=$11 --gpu=$12
```
- The explanation of hyper-parameters can be found in './code/FM_NFM/main.py'. 
- The well trained models are provided in './code/FM_NFM/best_models'. We have tuned the hyper-parameters and chosen the best ones.

### UCI Inference

#### item-feature controls
- Reranking
```
cd item_controls/Reranking
python UCI_reranking.py --model=FM --dataset=ml_1m
```

- Fine-grained and coarse-grained item-feature controls
```
cd item_controls/fine_coarse_UCI
python C_UCI_inference.py --model=FM --dataset=ml_1m
python F_UCI_inference.py --model=NFM --dataset=amazon_book
```

#### user-feature coarse-grained controls
- UCI and maskUF for FM and NFM
```
cd user_coarse_controls
python UCI_coarse_user_control.py --model=FM
```

- Inference for vanilla FM and NFM
```
python FM_NFM_inference.py --model=FM
```
Note that we only use DIGIX for the experiments of user-feature controls.

#### user-feature fine-grained controls
- UCI and changeUF for FM and NFM
```
cd user_fine_controls
python UCI_fine_user_control.py --model=FM
```

- Inference for vanilla FM and NFM
```
python FM_NFM_inference.py --model=FM
```


## Acknowledgment

Thanks to the FM/NFM implementation:

- [NFM-torch](https://github.com/guoyang9/NFM-pyorch/) from Yangyang Guo.
- [NFM-tensorflow](https://github.com/hexiangnan/neural_factorization_machine) from Xiangnan He. 

## License

NUS © [NExT++](https://www.nextcenter.org/)
