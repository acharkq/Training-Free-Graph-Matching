# README

This is the code for paper *[Training Free Graph Neural Networks for Graph Matching](https://arxiv.org/pdf/2201.05349.pdf)*. arXiv preprint.

## Description

**T**raining **F**ree **G**raph **M**atching (**TFGM**) is a framework to boost the performance of GNNs for graph matching without training. This github repository contains our exemplar implementations of TFGM with the popular [GraphSAGE](https://github.com/williamleif/GraphSAGE), [SplineCNN](https://github.com/rusty1s/pytorch_spline_conv), and [DGMC](https://github.com/rusty1s/deep-graph-matching-consensus). The idea is easy to implement and you can also try TFGM with other GNNs.

## Dependencies

1. PyTorch
2. [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric) 1.7.0

## Datasets

Download the DBP15k and the PPI dataset from this [Onedrive Link](https://1drv.ms/u/s!AuQRz5abAH5T7mW2VuUCVsUJW-hd?e=nRn2T5). Unzip the file in the `data` folder.

PascalVOC will be downloaded automatically when running codes.

## Reproduce Results in Paper

**PascalVOC**

```bash
>> python pascal.py --use_splinecnn --use_knn --use_dgmc --gpu_id 0
```

**DBP15k**

```bash
>> python dbp15k.py --dataset zh_en --use_dgmc --use_supervision --weight_free --gpu_id 0 ## Chinese-English KG pair
>> python dbp15k.py --dataset ja_en --use_dgmc --use_supervision --weight_free --gpu_id 0 ## Japanese-English KG pair
>> python dbp15k.py --dataset fr_en --use_dgmc --use_supervision --weight_free --gpu_id 0 ## French-English KG pair
```
    
**PPI**

```bash
>> python ppi.py --dataset extra_edge --use_dgmc --num_steps 100  --weight_free --rnd_dim 128 --gpu_id 0  ## Low-Conf Edge dataset
>> python ppi.py --dataset rewirement --use_dgmc --num_steps 100  --weight_free --rnd_dim 128 --gpu_id 0  ## Random Rewirement dataset
```

If you have any questions regarding running the code, please feel free to raise a github issue.

## Reference

If you use our code, please cite our paper

```bib
@article{liu2022training,
  title={Training Free Graph Neural Networks for Graph Matching},
  author={Liu, Zhiyuan and Cao, Yixin and Feng, Fuli and Wang, Xiang and Shang, Xindi and Tang, Jie and Kawaguchi, Kenji and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:2201.05349},
  year={2022}
}
```
