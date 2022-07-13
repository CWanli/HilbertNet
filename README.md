# Efficient Point Cloud Analysis Using Hilbert Curve

 The source code of our work **"Efficient Point Cloud Analysis Using Hilbert Curve"**
![img|center](./framework.png)
## News
HilbertNet is accepted to ECCV 2022

### Requirements
- PyTorch >= 1.6 
- pandas
- numpy
- [spconv](https://github.com/traveller59/spconv) (tested with spconv==2.1.22 and cuda==11.5)

## Data Preparation
Download and unzip the given data into `data/` directory

## Training
1. train the network by running "sh train.sh"

## Acknowledgments
We thanks for the opensource codebases, [PointNet](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) [SO-Net](https://github.com/lijx10/SO-Net.git) and [spconv](https://github.com/traveller59/spconv)