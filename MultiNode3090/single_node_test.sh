#!/bin/bash

# Define the array of models and batchsizes
declare -a models=("densenet121" "vgg19" "resnet50")
declare -a batchsizes=(32 64 128)

# Loop over each model -> script1
for model in "${models[@]}"
do
    # Loop over each batchsize
    for batchsize in "${batchsizes[@]}"
    do
        echo "Running training for model: $model with batchsize: $batchsize"
        # Run the PyTorch DDP training script with the specified model and batchsize
        torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="192.168.50.187" --master_port=12334 profile_test.py --model=$model --batchsize=$batchsize
        # torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr="192.168.50.187" --master_port=12334 profile_test.py --model=$model --batchsize=$batchsize
    done
done

# Loop over each model -> script2
for model in "${models[@]}"
do
    # Loop over each batchsize
    for batchsize in "${batchsizes[@]}"
    do
        echo "Running training for model: $model with batchsize: $batchsize"
        # Run the PyTorch DDP training script with the specified model and batchsize
        torchrun --nproc_per_node=2 --nnodes=2 --master_addr="192.168.50.187" --master_port=12334 ddp_profile.py --model=$model --batchsize=$batchsize
    done
done


# script3
echo "Testing allreduce for pure model"
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 standalone_allreduce.py


# torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="192.168.50.187" --master_port=12334 profile_test.py --model="resnet50" --batchsize=32
# torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr="192.168.50.187" --master_port=12334 profile_test.py --model="resnet50" --batchsize=32