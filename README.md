# Llama from scratch

This repository contains a rather simple implementation of Meta's famous Llama large language model using Pytorch. The implementation is pretty straightforward, with limited dependencies. It exists for educational purposes as it doesn't care about training and inference optimization. Specifically, it doesn't use flash-attention CUDA kernels (because the goal here is to understand how LLM works in details) nor does it make use of distributed data parallelism (I don't have access of clusters of GPUs anyway).

This implementation includes some of the improvements from Llama 2:

- positional encoding applied the query and key projections for each head in the multi-head self attention,
- RMS pre-normalization in transformer blocks,
- SwiGLU activation function in the feed-forwards.

To make the implementation end-to-end, we train the model on a small dataset using SentencePiece tokenizer.

## Getting Started <a name = "getting_started"></a>

Clone the repository:

`git clone https://github.com/clabrugere/scratch-llm.git`

### Dependencies

The implementation only depends on Python, Pytorch and Sentencepiece:

- python 3.11
- pytorch 2.0.1
- sentencepiece 0.1.99

## Usage <a name = "usage"></a>

You can refer to the notebook in `example/shakespeare.ipynb` for an end-to-end pre-training example on a small dataset, with a model of around 10M parameters only. The loss still decreases after 2000 iterations and the training loss is quite stable so we could train it further to improve its generative abilities.

<p align="center"><img src="resources/example-loss.png?raw=true"/></p>

## What is Llama

Let's go over some of the implementation details behind the large language model in this repository.

### Transformer and attention mechanism

The power of modern LLMs lies on chaining transformer blocks. A transformer block is composed of a succession of a positional encoding layer, a multi-head self-attention layer and a feed-forward network:

```
TransformerBlock(
  (norm_1): RMSNorm()
  (multihead_attn): MultiHeadAttention(
    (positional_encoding): RotaryPositionalEncoding()
    (proj_q): Linear(in_features=128, out_features=128, bias=False)
    (proj_k): Linear(in_features=128, out_features=128, bias=False)
    (proj_v): Linear(in_features=128, out_features=128, bias=False)
    (proj_out): Linear(in_features=128, out_features=128, bias=False)
  )
  (norm_2): RMSNorm()
  (feed_forward): FeedForward(
    (_layers): Sequential(
      (0): Linear(in_features=128, out_features=128, bias=False)
      (1): SwiGLU(
        (linear): Linear(in_features=128, out_features=256, bias=True)
      )
      (2): Linear(in_features=128, out_features=128, bias=False)
    )
  )
)
```

Chaining such blocks allow to learn more and more abstract representations of the input the deeper we go, but always with this weighted linear decomposition mechanism using the entire context.

#### Positional encoding

The positional encoding layer allows to make the self-attention layers "aware" of the relative or absolute position of each element in the input sequence. It is critical for modeling data where order is important (such as sentences) because the self-attention mechanism is position-invariant.

The original implementation of the Transformer as described in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) used trigonometric functions in their cosine positional encoding, described as:

```math
P(m, 2k) = sin( \frac{2m}{10000^{\frac{2k}{d*{emb}}}} )
```

```math
P(m, 2k+1) = cos( \frac{2m}{10000^{\frac{2k}{d*{emb}}}} )
```

<p align="center"><img src="resources/cosine_positional_encoding.png?raw=true"/></p>

Llama 2 instead use a recent mechanism to encode position directly into self-attention query and key projections, named [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864). It is specifically designed to have special properties to encode positional information: let $`f( x_i, p)`$ be the function encoding position of token $`x`$ at position $`p`$, then the inner product of the positional encoding of two tokens at different positions $`p_x`$ and $`p_y`$ has the property:

```math
\langle f( x_i, p_x), f( y_i, p_y) \rangle = f(x_i y_i, p_x - p_y)
```

Which means that two pairs of tokens with the same relative distance from each other will have the same dot product, satisfying the relative position property. It stays true within the self-attention as the transformation is done on the token vectors projected into the query and key spaces instead of on the raw vectors like the original Transformer. Intuitively, rotary embeddings are rotations of the query or key vectors by m times a fixed angle, where m is the position of the token in the sequence, which is the reason behind the relative positioning property. Independently of their absolute positions, their relative angle will be $`p_x \theta - p_y \theta = (p_x - p_y) \theta`$ where $`\theta`$ is constant, thus making their dot product the same. Another interesting property is that tokens far away in the sequence will have a greater angle between them after rotary embedding, making the dot product smaller, which make sense intuitively because we would expect tokens to be more independant the farther away they are from each others.

#### Multi-head self-attention

The multi-head self attention block learns - for a given element of the sequence - a linear combination of all the elements of the input projected into a "value" space, weighted by a similarity score between the projections in the "query" space and "key" space. In essence, it decomposes each elements of the input sequence into a linear combination of all the elements (including itself) where the weights are similarities that are learned through the projections. In the classic attention, the similarity metric is the dot-product.

For NLP, this mechanism allows to encode the importance of the other tokens within the context, for a given token. To allow for parallel training, we impose the constraint that a given token can only be decomposed into a weighted sum of all the previous tokens and itself but not the tokens coming afterward (it is the concept of causal masking: every token in a sequence is a training example, so the attention processes the whole sequence at once). Another way to understand what the "causal mask" is, is to see it as a sequence padding mechanism.

A clear advantage over previous approaches is that the whole context is accessible to a given token. The multi-head part is a way to learn different weights at the same time and thus provide different representations of the input sequence, allowing the model to be more expressive.

<p align="center"><img src="resources/multi-head-attention-init.png?raw=true"/></p>

#### Feed-forward

It is a simple MLP chaining of a linear map, a non-linear activation function and another linear map. As the output of the multi-head self-attention layer is a 3D tensor of shape (batch size, sequence length, token embedding dimensions), it is applied only on the last dimension.

### SwiGLU activation function

In the transformer architecture, the output of the attention layer pass through a simple multi layer perceptron. The most used activation function, ReLu - that simply zeroes negative values of its input - has been widely used but some variants have been proposed to improve model stability and convergence.

In [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288), authors mention that they used the SwiGLU variant to train their large language model. It is defined as:

```math
\text{SwiGLU}(\textbf{x}) = \text{Swish}(\textbf{x}W + b) \otimes (\textbf{x}V + c)
```

Authors of [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) empirically show that some flavors of a special class of activation functions named Gated Linear Units (GLU) provide some improvements over the regular Relu. In particular they slightly update the formulation of SwiGLU defined above:

```math
\text{SwiGLU}(\textbf{x}) = (\text{Swish}(\textbf{x}W_1) \otimes \textbf{x}V )W_2
```

In this implementation, we stick to the regular SwiGLU for simplicity.

### RMSNorm layer

A custom in the field of NLP was to use layer normalization to help improve training stability by centering and scaling the input's distribution and provide some robustness against noise because it makes its output invariant to offset and rescaling. Nevertheless it introduces a computational overhead for large and deep networks through the need to calculate input's mean and standard deviation.

Authors of [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) argue that the real advantage of layer normalization lies in the rescaling invariance rather than the offset invariance and propose to simply rescale the inputs using the root mean square statistic $`\text{RMS}(\textbf{x}) = \sqrt{ \frac{1}{n} \sum x_i }`$, applied before the activation function:

```math
\text{RMSNormLayer}(\textbf{x}) = \frac{\textbf{x}}{\text{RMS}(\textbf{x})} \textbf{g}
```

where $`\textbf{g}`$ is a learned parameter.

### Pre-training

The model learn a latent representation of the language in a self-supervised way with a surprisingly simple approach: given a large corpus, sequences of fixed size are sampled randomly to build the batched context as input of the model, and the targets are those very same sequences shifted by one element so that the model learn to predict the next token, given a context, by minimizing the cross-entropy through gradient descent:

```math
f_{\theta} = \text{argmin} \ L(X, Y)
```

```math
\textbf{x}_i = [t_0, ...t_{T}]_{i}
```

```math
\textbf{y}_i = [t_1, ..., t_{T + 1}]_{i}
```

<p align="center"><img src="resources/decoder-only.png?raw=true"/></p>

## References

- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [SentencePiece](https://github.com/google/sentencepiece)
