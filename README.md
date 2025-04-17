# FairMatch
This is the code of our KDD24 paper, titled "FairMatch: Promoting Partial Label Learning by Unlabeled Samples".

## Usage:
### CIFAR-10 ($\rho$=10%, q=0.1):

```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --rho 0.1 --partial_rate 0.1
```

### CIFAR-100 ($\rho$=20%, q=0.05):

```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --rho 0.2 --partial_rate 0.05 --warm_up 50
```

### CIFAR-100H ($\rho$=10%, q=0.3):

```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --rho 0.1 --partial_rate 0.3 --warm_up 50 --hierarchical
```

### STL-10 (q=0.1):

```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset stl10 --rho 1 --partial_rate 0.1
```
