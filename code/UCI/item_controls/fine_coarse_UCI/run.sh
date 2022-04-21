nohup python -u main.py --dataset=$1 --layers=$2 --lr=$3 --batch_size=$4 --dropout=$5 --lamda=$6 --optimizer=$7 --act_function=$8 --batch_norm=$9 --epochs=$10 --log_name=$11 --gpu=$12  > ./log/MLP_$1_$2layer_$3lr_$4bs_$5dropout_$6lamda_$7_$8_$9bn_$10epoch_$11.txt 2>&1 &
## sh run.sh ml_1m [32] 0.05 1024 0.3 0.1 Adagrad sigmoid 1 2000 log 0 
## sh run.sh amazon_book_only_first [32] 0.05 1024 0.3 0.1 Adagrad sigmoid 1 2000 log 0 
