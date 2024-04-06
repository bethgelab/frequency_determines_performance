# Pretraining Concept Frequency determines Multimodal Model Performance [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-360/) [![PyTorch](https://img.shields.io/badge/PyTorch-grey.svg?logo=PyTorch)](https://pytorch.org/blog/pytorch-1.9-released/) [![Paper](http://img.shields.io/badge/paper-arxiv.2211.16198-B31B1B.svg)](https://arxiv.org/abs/2211.16198)

This is the official codebase for the paper, "No Zero-Shot Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance".
Authors: [Vishaal Udandarao*](http://vishaal27.github.io/), [Ameya Prabhu*](https://drimpossible.github.io/), [Adhiraj Ghosh](https://adhirajghosh.github.io/), [Yash Sharma](https://www.yash-sharma.com/), [Philip H.S. Torr](https://scholar.google.com/citations?user=kPxa2w0AAAAJ&hl=en), [Adel Bibi](https://www.adelbibi.com/), [Samuel Albanie](http://samuelalbanie.com/) and [Matthias Bethge](https://scholar.google.com/citations?user=0z0fNxUAAAAJ). 

## Getting started
All our code was tested on Python 3.8.13 with Pytorch 2.0.1+cu117. Ideally, most of our scripts require access to a single GPU (uses `.cuda()` for inference). Inference can also be done on CPUs with minimal changes to the scripts.

#### Setting up environments
We recommend setting up a python virtual environment and installing all the requirements. Please follow these steps to set up the project folder correctly:

```bash
git clone https://github.com/bethgelab/frequency_determines_performance.git
cd frequency_determines_performance

conda create --name env python=3.8 -y
conda activate env
pip install -r requirements.txt
```

#### Setting up datasets
We provide detailed instructions on how to set up both pretraining and downstream test datasets in [`data/README.md`](https://github.com/bethgelab/frequency_determines_performance/blob/main/data/README.md).
