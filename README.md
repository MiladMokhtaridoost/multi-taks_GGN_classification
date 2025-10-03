GNN-Transformer for Hi-C Data Classification

This repository provides a PyTorch Geometric implementation of a GNN-Transformer pipeline for supervised classification of genomic Hi-C datasets. The model integrates graph neural networks (GNNs) with global pooling and linear layers to predict multiple biological labels from Hi-C contact maps.

The pipeline supports three multi-task outputs:

Health status: Healthy vs. Cancer (binary)

Tissue type: Six predefined tissue categories (multi-class)

Sex: Male vs. Female (binary)

It includes checkpointing for long jobs on HPC systems, GPU acceleration, and automatic recovery from interruptions.
