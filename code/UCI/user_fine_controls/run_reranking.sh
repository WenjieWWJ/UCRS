nohup python -u reranking.py --model=$1 --dataset=$2 --file_head=$3 >> new_inference_res.txt 2>&1 &
# python -u reranking.py --model=FM --dataset=ml_1m --file_head=FM_ml_1m_64hidden_[32]layer_0.05lr_1024bs_[0.3,0.3]dropout_0.1lamda_1bn_1500epoch
# python -u reranking.py --model=FM --dataset=amazon_book_only_first --file_head=FM_amazon_book_only_first_64hidden_[64]layer_0.05lr_1024bs_[0.5,0.2]dropout_0.1lamda_1bn_1000epoch
# python -u reranking.py --model=NFM --dataset=ml_1m --file_head=NFM_ml_1m_64hidden_[16]layer_0.01lr_1024bs_[0.3,0.3]dropout_0.1lamda_1bn_500epoch