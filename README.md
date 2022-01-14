# README

This is the code for paper *Training Free Graph Neural Networks for Graph Matching*. arXiv preprint.

## Dependencies

1. PyTorch
2. [Pytorh Geometric](https://github.com/rusty1s/pytorch_geometric) 1.7.0

## Datasets

Download the DBP15k and the PPI dataset from this onedrive [link](https://1drv.ms/u/s!AuQRz5abAH5T7mW2VuUCVsUJW-hd?e=nRn2T5). Unzip them in the `data` folder.

PascalVOC will be downloaded automatically when running codes.

## Reproduce results in paper

**Pascal**

```bash
>> python pascal.py --use_splinecnn --use_knn --use_dgmc --gpu_id 0
```

**DBP15k**

```bash
>> python dbp15k.py --dataset zh_en --use_dgmc --use_supervision --weight_free --gpu_id 0
```
    
**PPI**

```bash
>> python ppi.py --dataset extra_edge --use_dgmc --num_steps 100  --weight_free --rnd_dim 128 --gpu_id 0
```

## Reference

```bib
@article{liu2022training,
  title={Training Free Graph Neural Networks for Graph Matching},
  author={Liu, Zhiyuan and Cao, Yixin and Feng, Fuli and Wang, Xiang and Shang, Xindi and Tang, Jie and Kawaguchi, Kenji and Chua, Tat-Seng},
  journal={arXiv preprint},
  year={2022}
}
```
