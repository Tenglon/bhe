CUDA_VISIBLE_DEVICES=1 python hyp_quick.py --n_bits 2 --emb_dim 64 > logs/hyp_quick.log
CUDA_VISIBLE_DEVICES=1 python hyp_quick.py --n_bits 2 --emb_dim 128 >> logs/hyp_quick.log
CUDA_VISIBLE_DEVICES=1 python hyp_quick.py --n_bits 2 --emb_dim 256 >> logs/hyp_quick.log

CUDA_VISIBLE_DEVICES=1 python hyp_quick.py --n_bits 4 --emb_dim 64 >> logs/hyp_quick.log
CUDA_VISIBLE_DEVICES=1 python hyp_quick.py --n_bits 4 --emb_dim 128 >> logs/hyp_quick.log
