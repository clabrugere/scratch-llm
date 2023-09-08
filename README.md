# Llama from scratch

This repository contains a rather simple implementation of Meta's famous Llama large language model using Pytorch. The implementation is pretty straightforward, with limited dependencies. It exists for educational purposes as it doesn't care about training and inference optimization. Specifically, it doesn't use flash-attention CUDA kernels (because the goal of this implementation is to understand how LLM works in details) nor does it make use of distributed data parallelism (I don't have access of clusters of GPUs anyway).

This implementation includes some of the improvements from Llama 2:

- positional encoding before every transformer block,
- RMS pre-normalization in transformer blocks (so before multi-head attention and feedforward),
- SwiGLU activation function

To make the implementation end-to-end, we train the model on a small dataset, using sentecepiece tokenizer.

## Getting Started <a name = "getting_started"></a>

Clone the repository:

`git clone <url>`

### Dependencies

The implementation only depends on Python and Pytorch:

- python 3.11
- pytorch 2.0.1
- sentencepiece 0.1.99

## Usage <a name = "usage"></a>

Add notes about how to use the system.

## What is Llama

Let's go over some of the implementation details behind Llama large language model.

### Transformer

### Self attention and multi-head attention

### SwiGLU activation function

In the transformer achitecture, the output of the attention layer pass through a simple multi layer perceptron: two linear layers with an activation function in between. The most used activation function, ReLu - that simply zeroes negative values of its input - has been widely used but some variants have been proposed to improve model stability and convergence.

In [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288), authors mention that they used the SwiGLU variant to train their large language model defined as $` \text{SwiGLU}(\textbf{x}) = \text{Swish}(\textbf{x}W + b) \otimes (\textbf{x}V + c)  `$.

Authors of [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) empirically show that some flavors of a special class of activation functions named Gated Linear Units (GLU) provide some improvements over the regular Relu. In particular they slightly update the formulation of SwiGLU defined above: $` \text{SwiGLU}(\textbf{x}) = (\text{Swish}(\textbf{x}W_1) \otimes \textbf{x}V )W_2 `$.

In this implementation, we stick to the regular SwiGLU for simplicity.

### RMSNorm layer

A custom in the field of NLP was to use layer normalization to help improve training stability by centering and scaling the its input's distribution and provide some rubustness againt noise because it is invariant to inputs offset and rescaling. Nevertheless it introduces a computational overhead for large and deep networks through the need to calculate input's mean and standard deviation.

Authors of [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) argue that the real advantage of layer normalization lies in the rescaling invariance rather than the offset invariance and propose to simply rescale the inputs using the root mean square statistic $`\text{RMS}(\textbf{x}) = \sqrt{ \frac{1}{n} \sum x_i }`$, applied before the activation function: $`\text{RMSNormLayer}(\textbf{x}) = \frac{\textbf{x}}{\text{RMS}(\textbf{x})} \textbf{g}`$ where $`\textbf{g}`$ is a learned parameter.

## References

- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
