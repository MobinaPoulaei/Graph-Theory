# Graph Neural Networks (GNNs) Implementation on Cora Dataset

## Overview

This repository contains the implementation of Graph Neural Networks (GNNs) using PyTorch and PyTorch Geometric. The main focus is on implementing two types of GNN models: Graph Convolutional Networks (GCNs) and Graph Attention Networks (GATs), applied to the Cora citation network dataset. The objective is to classify scientific papers into one of seven categories based on the structure and features of the citation network.

## Dataset

The Cora dataset is a commonly used benchmark in graph-based learning. It consists of 2,708 nodes (papers) and 5,429 edges (citations). Each node is described by a word vector with 1,433 features and is labeled with one of seven classes representing the paper's topic.

## Models

### 1. Graph Convolutional Networks (GCNs)

GCNs extend the concept of convolution to graph-structured data. The GCN model implemented here consists of two layers:
- **GCN Layer 1**: Transforms the input features into a hidden representation.
- **GCN Layer 2**: Maps the hidden representation to the final class scores.

### 2. Graph Attention Networks (GATs)

GATs introduce an attention mechanism to GNNs, allowing the model to assign different importance to each neighboring node during feature aggregation.
- **GAT Layer 1**: Uses multiple attention heads to transform input features into a hidden representation.
- **GAT Layer 2**: Aggregates these features and maps them to class scores.

For more detailed explanations, visit the full article on Medium: [Introduction to Graph Neural Networks](https://medium.com/p/1f96126cb727/)
