nohup python -u main.py --model=$1 --dataset=$2 --hidden=$3 --layers=$4 --lr=$5 --batch_size=$6 --dropout=$7 --lamda=$8 --batch_norm=$9 --epochs=$10 --log_name=$11 --gpu=$12  > ./log/$1_$2_$3hidden_$4layer_$5lr_$6bs_$7dropout_$8lamda_$9bn_$10epoch_$11.txt 2>&1 &
## sh run.sh FM ml_1m 64 [32] 0.05 1024 [0.3,0.3] 0.1 1 1500 0
## sh run.sh FM amazon_book 64 [64] 0.05 1024 [0.5,0.2] 0.1 1 1000 0