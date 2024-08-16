---
title: "Overview of distributed training technologies for Transformers"
categories:
  - distributed-training
tags:
  - llms
  - deep learning
  - deepspeed
  - pytorch
  - transformers
layout: single
classes: wide
toc: true
toc_label: Table of Contents
toc_sticky: true
---

# Why do distributed training?
Current transformers based large language models have tens or hundreds of billions of parameters. Training such models on a single machine is often infeasible due to memory and computational limitations. Best available Nvidia A100 GPU has only 80 GB of high bandwidth memory, while one of the best open source llm Llama3 70B model occupies around 140 GB. These models are also trained on very large datasets containing trillions of tokens, and this requires significant amount of compute which results in very long training time. By distributed training of such models across multiple gpus the workload can be parallelized significantly reducing the training time.

# Different distributed training strategies
- Distributed Data Parallel
- Zero Optimizer
- Model Parallelism
- Pipeline Parallelism
- Pytorch's Fully Sharded Data Parallel


## Distributed Data Parallel
Most common strategy for doing parallel training.

## ZerO Optimizer
A special optimizer that dissects optimizer state's to multiple devices.

## Model Parallelism
First introduced in Megatron paper by Nvidia

## Pipeline Parallelism
Performs parallelism by dissecting models horizontally

## PyTorch's FSDP
New distributed training strategy introduced in PyTorch framework.

