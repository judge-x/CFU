# Heterogeneous Federated Unlearning with Fine-tuned Contrastive Learning
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



# Client-level

# FMNIST
pre-train
```
python main.py --gpuIndex 0 --dataset fmnist --batchsize 256  --localEpochs 1 --lr 0.001 --globalEpochs 200 --iid --alpha 0.5 --numClient 10
```

unlearning
```
python main.py --gpuIndex 0 --dataset fmnist --batchsize 256 --localEpochs 1 --lr 0.001 --globalEpochs 200  --numClient 10 --iid --alpha 1 --unlearningChoice --unlearningEpochs 20 --unlearningMethod cont --tau7 --mu 0.1 --gamme 0.5
```



