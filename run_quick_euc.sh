CUDA_VISIBLE_DEVICES=0 python euc_quick.py --n_bits 2 --emb_dim 64 > logs/euc_quick.log
CUDA_VISIBLE_DEVICES=0 python euc_quick.py --n_bits 2 --emb_dim 128 >> logs/euc_quick.log
CUDA_VISIBLE_DEVICES=0 python euc_quick.py --n_bits 2 --emb_dim 256 >> logs/euc_quick.log

CUDA_VISIBLE_DEVICES=0 python euc_quick.py --n_bits 4 --emb_dim 64 >> logs/euc_quick.log
CUDA_VISIBLE_DEVICES=0 python euc_quick.py --n_bits 4 --emb_dim 128 >> logs/euc_quick.log
