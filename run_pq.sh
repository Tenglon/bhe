python pq.py --dataset cifar100 --M 4 --n_bits 8 > logs/pq_cifar100.log
python pq.py --dataset cifar100 --M 8 --n_bits 8 >> logs/pq_cifar100.log
python pq.py --dataset cifar100 --M 16 --n_bits 8 >> logs/pq_cifar100.log
python pq.py --dataset cifar100 --M 32 --n_bits 8 >> logs/pq_cifar100.log
python pq.py --dataset cifar100 --M 64 --n_bits 8 >> logs/pq_cifar100.log

python pq.py --dataset cifar100 --M 4 --n_bits 4 >> logs/pq_cifar100.log
python pq.py --dataset cifar100 --M 8 --n_bits 4 >> logs/pq_cifar100.log
python pq.py --dataset cifar100 --M 16 --n_bits 4 >> logs/pq_cifar100.log
python pq.py --dataset cifar100 --M 32 --n_bits 4 >> logs/pq_cifar100.log
python pq.py --dataset cifar100 --M 64 --n_bits 4 >> logs/pq_cifar100.log
python pq.py --dataset cifar100 --M 128 --n_bits 4 >> logs/pq_cifar100.log

python pq.py --dataset imagenet --M 4 --n_bits 8 > logs/pq_imnet.log
python pq.py --dataset imagenet --M 8 --n_bits 8 >> logs/pq_imnet.log
python pq.py --dataset imagenet --M 16 --n_bits 8 >> logs/pq_imnet.log
python pq.py --dataset imagenet --M 32 --n_bits 8 >> logs/pq_imnet.log
python pq.py --dataset imagenet --M 64 --n_bits 8 >> logs/pq_imnet.log

python pq.py --dataset imagenet --M 4 --n_bits 4 >> logs/pq_imnet.log
python pq.py --dataset imagenet --M 8 --n_bits 4 >> logs/pq_imnet.log
python pq.py --dataset imagenet --M 16 --n_bits 4 >> logs/pq_imnet.log
python pq.py --dataset imagenet --M 32 --n_bits 4 >> logs/pq_imnet.log
python pq.py --dataset imagenet --M 64 --n_bits 4 >> logs/pq_imnet.log
python pq.py --dataset imagenet --M 128 --n_bits 4 >> logs/pq_imnet.log

