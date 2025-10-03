# GNN-Transformer for Hi-C Data Classification

This repository provides a PyTorch Geometric implementation of a **GNN-Transformer** pipeline for supervised classification of genomic Hi-C datasets. The model integrates graph neural networks (GNNs) with global pooling and linear layers to predict multiple biological labels from Hi-C contact maps.

The pipeline supports **three multi-task outputs**:
- **Health status**: Healthy vs. Cancer (binary)
- **Tissue type**: Six predefined tissue categories (multi-class)
- **Sex**: Male vs. Female (binary)

It includes checkpointing for long jobs on HPC systems, GPU acceleration, and automatic recovery from interruptions.

---

## Features
- Converts **Hi-C contact files** into graph representations (nodes = genomic bins, edges = contact frequencies).
- **Multi-task learning** with three parallel classification heads.
- **Checkpointing per epoch** to support HPC walltime limits.
- Automatic resumption from the latest checkpoint.
- **Evaluation module** reporting classification accuracies for all tasks.

---

## Requirements
- Python ≥ 3.10  
- [PyTorch](https://pytorch.org/) with CUDA support  
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)  
- pandas, numpy  

Cluster users can load via modules (example):  
```bash
module load python/3.11.3_torch_gpu
```

---

## Input Data
- 1. **Hi-C edge list files**
Each dataset is a tab-delimited text file with the following columns:

```
chrA_ID   chrB_ID   frequency
1_82      1_90      12.5
```

- chrX_binIndex format for bin identifiers
- Frequencies are floating-point contact strengths

- 2. **Label table (CSV)**
     Example:

```
     SRX_ID,Healthy/Cancer,Tissue,Sex
     SRX001,Healthy,Brain,M
     SRX002,Cancer,Lung,F
```
