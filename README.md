# UCRS

> This is the code of "User-controllable Recommendation Against Filter Bubbles".

## Environment

- Anaconda 3
- python 3.7.3
- pytorch 1.4.0
- numpy 1.16.4

## Usage

### Data

The three datasets are released in the './data' folder.

### Code

The code for training and inference is in the './code' folder. FM and NFM are first trained, and then UCI is used for inference. 

#### FM and NFM training
```
python main.py --model=$1 --dataset=$2 --hidden=$3 --layers=$4 --lr=$5 --batch_size=$6 --dropout=$7 --lamda=$8 --batch_norm=$9 --epochs=$10 --log_name=$11 --gpu=$12
```
- The explanation of hyper-parameters can be found in './code/FM_NFM/main.py'. 
- The best hyper-parameter settings are presented in './code/FM_NFM/best_models'.

### UCI Inference

#### item-feature controls
```
cd item_controls
python inference.py 
```
#### user-feature coarse-grained controls
- UCI and maskUF for FM and NFM
```
cd user_coarse_controls
python UCI_coarse_user_control.py --model=FM
```
- inference for vanilla FM and NFM
```
python FM_NFM_inference.py --model=FM
```
#### user-feature fine-grained controls
- UCI and changeUF for FM and NFM
```
cd user_fine_controls
python UCI_fine_user_control.py --model=FM
```
- inference for vanilla FM and NFM
```
python FM_NFM_inference.py --model=FM
```
