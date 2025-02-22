# train

python train.py --dataset METR-LA --batch_size 64 --epochs 100 --clip 5 --n_experts 3 --n_stacks 2 --time_0 0.9 --step_0 0.9 --time_1 0.9 --step_1 0.3 --time_2 0.9 --step_2 0.3 --end_dim 128 --decoder_types 1,2

python train.py --dataset PEMS-BAY --batch_size 64 --epochs 100 --milestones 25,50,75 --clip 5 --n_experts 3 --n_stacks 3 --time_0 0.9 --step_0 0.9 --time_1 0.9 --step_1 0.3 --time_2 0.9 --step_2 0.3 --decoder_types 1,1

python train.py --dataset PEMS03 --batch_size 64 --epochs 300 --clip 5 --in_dim 3 --n_experts 3 --n_stacks 1 --time_0 0.9 --step_0 0.9 --time_1 1.0 --step_1 0.2 --time_2 0.9 --step_2 0.45 --decoder_types 1,2 --end_dim 128 --lr 0.001 --wdecay 0.0001 --dropout 0.3

python train.py --dataset PEMS04 --batch_size 64 --epochs 300 --clip 5 --in_dim 3 --n_experts 3 --n_stacks 2 --time_0 0.9 --step_0 0.9 --time_1 0.9 --step_1 0.3 --time_2 1.0 --step_2 0.2 --decoder_types 1,1 --end_dim 64 --lr 0.001 --wdecay 0.0001 --dropout 0.1

# test

python test.py --dataset METR-LA --start 99 --n_experts 3 --n_stacks 2 --time_0 0.9 --step_0 0.9 --time_1 0.9 --step_1 0.3 --time_2 0.9 --step_2 0.3 --end_dim 128 --decoder_type 1,2

python test.py --dataset PEMS-BAY --start 99 --n_experts 3 --n_stacks 3 --time_0 0.9 --step_0 0.9 --time_1 0.9 --step_1 0.3 --time_2 0.9 --step_2 0.3 --decoder_types 1,1

python test.py --dataset PEMS03 --start 299 --batch_size 64 --in_dim 3 --n_experts 3 --n_stacks 1 --time_0 0.9 --step_0 0.9 --time_1 1.0 --step_1 0.2 --time_2 0.9 --step_2 0.45 --decoder_types 1,2 --end_dim 128 --dropout 0.3

python test.py --dataset PEMS04 --start 297 --in_dim 3 --n_experts 3 --n_stacks 3 --time_0 0.9 --step_0 0.9 --time_1 0.9 --step_1 0.3 --time_2 1.0 --step_2 0.2 --decoder_types 1,2 --end_dim 128 --dropout 0.1