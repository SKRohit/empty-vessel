---
title: "Intro to DeepSpeed"
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

# What is DeepSpeed?
Current transformers based large language models have tens or hundreds of billions of parameters. Training such models on a single machine is often infeasible due to memory and computational limitations. Best available Nvidia A100 GPU has only 80 GB of high bandwidth memory, while one of the best open source llm Llama3 70B model occupies around 140 GB. These models are also trained on very large datasets containing trillions of tokens, and this requires significant amount of compute which results in very long training time. By distributed training of such models across multiple gpus the workload can be parallelized significantly reducing the training time.

# Building blocks of distributed training in DeepSpeed
- Distributed Data Parallel
- Zero Optimizer
- Model Parallelism
- Pipeline Parallelism
- Pytorch's Fully Sharded Data Parallel

## ZerO in DeepSpeed
A special optimizer that dissects optimizer state's to multiple devices.

## Model Parallelism in DeepSpeed
First introduced in Megatron paper by Nvidia

## Pipeline Parallelism in DeepSpeed
Performs parallelism by dissecting models horizontally