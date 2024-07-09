# Model-Contrastive Federated Learning
This is the code for paper 

## Citation

## Backdoor(data-level):

# FMNIST:
pre-train
```
python mainBackdoor.py --gpuIndex 2 --dataset fmnist --batchsize 256  --localEpochs 2 --backdoor --backRate 0.64 --lr 0.001 --globalEpochs 80 --numClient 10
```
unlearning
```
python mainBackdoor.py --gpuIndex 2 --dataset fmnist --batchsize 256  --localEpochs 2 --backdoor --backRate 0.64 --lr 0.001 --globalEpochs 80 --numClient 10 --unlearningChoice --unlearningMethods='cont' --unlearningEpochs 20
```

# EMNIST
pre-train:
'''
python mainBackdoor.py --gpuIndex 1 --dataset emnist --batchsize 64  --localEpochs 1 --backdoor --backRate 0.5 --lr 0.005 --globalEpochs 50 --numClient 10
'''

unlearning
```
python mainBackdoor.py --gpuIndex 1 --dataset emnist --batchsize 64  --localEpochs 1 --backdoor --backRate 0.5 --lr 0.005 --globalEpochs 50 --numClient 10 --unlearningChoice --unlearningMethods cont --temperature 0.1 --mu 0.01
```

# CIFAR10
pretrain
```
python mainBackdoor.py --gpuIndex 3 --dataset cifar10 --batchsize 128  --localEpochs 1 --backdoor --backRate 0.75 --lr 0.001 --globalEpochs 50 --numClient 1
```

unlearning
```

```


# Client-level

# Cifar10
pre-train
```
python main.py --gpuIndex 1 --dataset cifar10 --batchsize 512 --localEpochs 5 --lr 0.001 --globalEpochs 100 --iid --alpha 0.5 --numClient 10 --embeddingLen 128
```
```
python main.py --gpuIndex 3 --dataset cifar10 --batchsize 512 --localEpochs 5 --lr 0.001 --globalEpochs 100 --iid --alpha 1 --numClient 10 --embeddingLen 128 --unlearningChoice --unlearningEpochs 50 --unlearningMethod cont --tau 0.1 --mu 0.1 --gamme 0.5
```

# EMNIST
python main.py --gpuIndex 2 --dataset emnist --batchsize 256  --localEpochs 1 --lr 0.001 --globalEpochs 100 --iid --alpha 0.1 --numClient 10

python main.py --gpuIndex 2 --dataset emnist --batchsize 256  --localEpochs 1 --lr 0.001 --globalEpochs 100  --numClient 10 --iid --alpha 0.1 --unlearningChoice --unlearningEpochs 20 --unlearningMethods cont --tau 0.07 --mu 0.1 --gamme 0.2 -fe 5

python main.py --gpuIndex 3 --dataset emnist --batchsize 256 --localEpochs 1 --lr 0.001 --globalEpochs 100  --numClient 10 --iid --alpha 0.05 --unlearningChoice --unlearningEpochs 30 --unlearningMethod cont --tau 0.7 --mu 1 --gamme 0.1 (best)



# FMNIST
pre-train
```
python main.py --gpuIndex 0 --dataset fmnist --batchsize 256  --localEpochs 1 --lr 0.001 --globalEpochs 200 --iid --alpha 0.5 --numClient 10
```

unlearning
```
python main.py --gpuIndex 2 --dataset fmnist --batchsize 256 --localEpochs 1 --lr 0.001 --globalEpochs 200  --numClient 10 --iid --alpha 0.1 --unlearningChoice --unlearningEpochs 50 --unlearningMethod cont --tau 0.0001 --mu 0.1
```
```
python main.py --gpuIndex 2  --dataset fmnist --batchsize 256 --localEpochs 1 --lr 0.001 --globalEpochs 200  --numClient 10 --iid --alpha 0.3 --unlearningChoice --unlearningEpochs 30 --unlearningMethod cont --tau 0.007 --mu 0.1 --gamme 1
```

python main.py --gpuIndex 2  --dataset fmnist --batchsize 256 --localEpochs 1 --lr 0.001 --globalEpochs 200  --numClient 10 --iid --alpha 0.05 --unlearningChoice --unlearningEpochs 20 --unlearningMethod cont --tau 1 --mu 1 --gamme 0.001


