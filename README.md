# Llama from scratch

This repository contains a rather simple implementation of Meta's famous Llama large language model using Pytorch. The implementation is pretty straightforward, with limited dependencies. It exists for educational purposes as it doesn't care about training and inference optimization. Specifically, it doesn't use flash-attention CUDA kernels (because the goal here is to understand how LLM works in details) nor does it make use of distributed data parallelism (I don't have access of clusters of GPUs anyway).

This implementation includes some of the improvements from Llama 2:

- rotary positional encoding applied to the query and key projections for each head in the multi-head self-attention,
- RMS pre-normalization in transformer blocks,
- SwiGLU activation function in the feed-forwards.

To make the implementation end-to-end, we train the model on a small dataset using a custom BPE tokenizer.

## Getting Started <a name = "getting_started"></a>

Clone the repository:

`git clone https://github.com/clabrugere/scratch-llm.git`

### Dependencies

The implementation only depends on Python and Pytorch:

- python 3.11
- pytorch 2.6+

## Usage <a name = "usage"></a>

You can refer to the notebook in `example/shakespeare.ipynb` for an end-to-end pre-training example on a small dataset, with a model of around 3M parameters only trained over 5000 steps of random batches sampled from `tinyshakespear.txt`

<p align="center"><img src="resources/example-loss.png?raw=true"/></p>

As we appended `<eos>` at each prose block boundaries, the model has learned to generate proses if prompted with a character's label:

```python
prompt = tokenizer.encode("KING HENRY VI:\n")
inputs = torch.tensor(prompt, dtype=torch.int32).unsqueeze(0).to(train_config.device)
out = model.generate(inputs, max_seq_len=256, stop_tokens={eos_id})
print(tokenizer.decode(out.tolist(), skip_special_tokens=True))
```

Generates

```
KING HENRY VI:
And in our France, Lord I do stay, I confess
That you presently laid strong as we lends,
To bear this fell for mind to great your house:
So twast the first Sir Smile, fast a sus.
```

## What is Llama

Let's go over some of the implementation details behind the large language model in this repository.

### Transformer and attention mechanism

The power of modern LLMs lies in chaining transformer blocks. A transformer block is composed of a multi-head self-attention layer (which applies rotary positional encoding internally to the query and key projections) followed by a feed-forward network, each preceded by a pre-normalization layer:

```
TransformerBlock(
  (norm_attn): RMSNorm()
  (multihead_attn): MultiHeadAttention(
    (positional_encoding): RotaryPositionalEncoding()
    (proj_qkv): Linear(in_features=128, out_features=384, bias=False)
    (proj_out): Linear(in_features=128, out_features=128, bias=False)
  )
  (norm_ffn): RMSNorm()
  (feed_forward): FeedForward(
    (0): Linear(in_features=128, out_features=128, bias=False)
    (1): SwiGLU(
      (linear): Linear(in_features=128, out_features=256, bias=True)
    )
    (2): Linear(in_features=128, out_features=128, bias=False)
  )
)
```

Chaining such blocks allow to learn more and more abstract representations of the input the deeper we go, but always with this weighted linear decomposition mechanism using the entire context.

#### Positional encoding

The positional encoding layer allows to make the self-attention layers "aware" of the relative or absolute position of each element in the input sequence. It is critical for modeling data where order is important (such as sentences) because the self-attention mechanism is position-invariant.

The original implementation of the Transformer as described in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) used trigonometric functions in their cosine positional encoding, described as:

```math
P(m, 2k) = sin( \frac{2m}{10000^{\frac{2k}{d_{emb}}}} )
```

```math
P(m, 2k+1) = cos( \frac{2m}{10000^{\frac{2k}{d_{emb}}}} )
```

<p align="center"><img src="resources/cosine_positional_encoding.png?raw=true"/></p>

Llama 2 instead uses a mechanism to encode position directly into self-attention query and key projections, named [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864). It is specifically designed to have special properties to encode positional information: let $`f( x_i, p)`$ be the function encoding position of token $`x`$ at position $`p`$, then the inner product of the positional encoding of two tokens at different positions $`p_x`$ and $`p_y`$ has the property:

```math
\langle f( x_i, p_x), f( y_i, p_y) \rangle = g(x_i, y_i, p_x - p_y)
```

Which means that two pairs of tokens with the same relative distance from each other will have the same dot product, satisfying the relative position property. It stays true within the self-attention as the transformation is done on the token vectors projected into the query and key spaces instead of on the raw vectors like the original Transformer. Intuitively, rotary embeddings are rotations of the query or key vectors by m times a fixed angle, where m is the position of the token in the sequence, which is the reason behind the relative positioning property. Independently of their absolute positions, their relative angle will be $`p_x \theta - p_y \theta = (p_x - p_y) \theta`$ where $`\theta`$ is constant, thus making their dot product the same. Another interesting property is that tokens far away in the sequence will have a greater angle between them after rotary embedding, making the dot product smaller, which makes sense intuitively because we would expect tokens to be more independent the farther away they are from each other.

#### Multi-head self-attention

The multi-head self attention block learns - for a given element of the sequence - a linear combination of all the elements of the input projected into a "value" space, weighted by a similarity score between the projections in the "query" space and "key" space. In essence, it decomposes each elements of the input sequence into a linear combination of all the elements (including itself) where the weights are similarities that are learned through the projections. In the classic attention, the similarity metric is the dot-product.

For NLP, this mechanism allows to encode the importance of the other tokens within the context, for a given token. To allow for parallel training, we impose the constraint that a given token can only be decomposed into a weighted sum of all the previous tokens and itself but not the tokens coming afterward (it is the concept of causal masking: every token in a sequence is a training example, so the attention processes the whole sequence at once).

A clear advantage over previous approaches is that the whole context is accessible to a given token. The multi-head part is a way to learn different weights at the same time and thus provide different representations of the input sequence, allowing the model to be more expressive.

<p align="center"><img src="resources/multi-head-attention-init.png?raw=true"/></p>

#### Feed-forward

It is a simple MLP chaining of a linear map, a non-linear activation function and another linear map. As the output of the multi-head self-attention layer is a 3D tensor of shape (batch size, sequence length, token embedding dimensions), it is applied only on the last dimension.

### SwiGLU activation function

In the transformer architecture, the output of the attention layer pass through a simple multi layer perceptron. The most used activation function, ReLU - that simply zeroes negative values of its input - has been widely used but some variants have been proposed to improve model stability and convergence.

In [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288), authors mention that they used the SwiGLU variant to train their large language model. It is defined as:

```math
\text{SwiGLU}(\textbf{x}) = \text{Swish}(\textbf{x}W + b) \otimes (\textbf{x}V + c)
```

Authors of [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) empirically show that some flavors of a special class of activation functions named Gated Linear Units (GLU) provide some improvements over the regular Relu. In particular they slightly update the formulation of SwiGLU defined above:

```math
\text{SwiGLU}(\textbf{x}) = (\text{Swish}(\textbf{x}W_1) \otimes \textbf{x}V )W_2
```

This implementation uses this extended formulation, where the down-projection $`W_2`$ is handled by the final linear layer of the feed-forward block.

### RMSNorm layer

A common practice in the field of NLP was to use layer normalization to help improve training stability by centering and scaling the input's distribution and provide some robustness against noise because it makes its output invariant to offset and rescaling. Nevertheless it introduces a computational overhead for large and deep networks through the need to calculate input's mean and standard deviation.

Authors of [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) argue that the real advantage of layer normalization lies in the rescaling invariance rather than the offset invariance and propose to simply rescale the inputs using the root mean square statistic $`\text{RMS}(\textbf{x}) = \sqrt{ \frac{1}{n} \sum x_i^2 }`$, applied before the activation function:

```math
\text{RMSNormLayer}(\textbf{x}) = \frac{\textbf{x}}{\text{RMS}(\textbf{x})} \textbf{g}
```

where $`\textbf{g}`$ is a learned parameter.

### BPE Tokenizer

Before a model can process text, it must be converted into a sequence of integer IDs. Byte Pair Encoding (BPE) is the algorithm used by most modern LLMs to build that vocabulary.

#### How BPE works

The vocabulary starts as 256 tokens, one per possible byte value. Because the input text is UTF-8 encoded at the byte level, the tokenizer can represent any string without out-of-vocabulary tokens.

Training then iteratively extends the vocabulary: find the most frequent adjacent pair of token IDs in the training corpus, assign it a new ID, replace every occurrence of that pair with the new ID, and repeat until `max_vocab_size` is reached. Each new token is the concatenation of the two bytes or token sequences it was merged from, so decoding is always a straightforward lookup. This allows to trade vocabulary size with compressed inputs (and so virtually increasing the context at fixed sequence length).

Encoding a new string applies the learned merges in the order they were learned: a merge with a lower ID was learned earlier (on a more frequent pair) and therefore has higher priority. Two adjacent tokens that can be merged are replaced by their merged ID, and the process repeats until no more merges apply.

#### Why a naive implementation is slow

A straightforward implementation of training re-scans the entire token sequence after every merge to recount all pair frequencies. With a training corpus of $n$ bytes and $V$ merges to learn, this is $O(nV)$, prohibitively slow for large vocabularies or large corpora.

Encoding has the same problem: applying each merge requires another full pass over the sequence.

#### `TokenSequence`: O(1) deletion via a doubly-linked list

When two adjacent tokens at positions $i$ and $i+1$ are merged into a single token, the right token must be removed from the sequence. Doing this in a plain array requires shifting every subsequent element, $O(n)$ per merge.

`TokenSequence` represents the sequence as a doubly-linked list backed by two plain integer arrays (`left[i]`, `right[i]`). Removing node $i+1$ is a constant-time pointer update: link $i$ directly to $i+2$, and link $i+2$ back to $i$. No data moves.

#### `PairIndex` — amortized O(log n) pair lookup via a lazy heap

Knowing which pair to merge next requires the most frequent pair across the entire sequence. Re-scanning from scratch each time costs $O(n)$ per step.

`PairIndex` maintains a max-heap of `(-count, pair)` entries alongside a `Counter` of live frequencies. The key insight is that each merge only affects two *boundary pairs*: if $(a, b) \to c$ is applied at position $i$, only the pair to the left of $i$ and the pair to the right of $i+1$ change — everything else in the sequence is untouched. After applying a merge, `PairIndex` removes the two stale boundary pairs and adds the two updated ones, each costing $O(\log n)$.

The heap is *lazy*: when a pair's count changes, a new heap entry is pushed rather than updating the existing one (it would cost a linear scan and re-heapify otherwise, so we trade this for a bit of additional memory instead). Stale entries (where the heap's stored count no longer matches the live `Counter`) are detected and skipped on pop. The heap is never explicitly cleaned up; stale entries are simply ignored as they surface.

Combined, these two structures reduce training from $O(nV)$ to $O(n \log n)$ and encoding from $O(nV)$ to $O(n \log V)$.

### Pre-training

The model learns a latent representation of the language in a self-supervised way with a surprisingly simple approach: given a large corpus, sequences of fixed size are sampled randomly to build the batched context as input of the model, and the targets are those very same sequences shifted by one element so that the model learns to predict the next token, given a context, by minimizing the cross-entropy through gradient descent:

```math
\theta^* = \text{argmin}_{\theta} \ L(X, Y)
```

```math
\textbf{x}_i = [t_0, ...t_{T}]_{i}
```

```math
\textbf{y}_i = [t_1, ..., t_{T + 1}]_{i}
```

<p align="center"><img src="resources/decoder-only.png?raw=true"/></p>

### Inference with KV caching

Naive inference with auto-regressive transformer models consists in generating one token at a time, append it the the context, and generate a token again (hence auto-regressive).

To generate token $t$, the model needs keys $[k_1, ..., k_{t-1}]$ and values $[v_1, ..., v_{t-1}]$ for all the previous positions to calculate attention scores with respect to the query $q_t$ corresponding to the token. Instead of recomputing those at each generating step, they can be cached into what is usually called a KV cache, drastically accelerating inference (but at the cost of a larger memory footprint). 

The cache simply stores a buffer of key and values for past positions and a pointer to the current sequence length. Generating with it becomes:

0. Prefill the cache with $K = X W_K$ and $V = X W_V$ where $X$ is the encoded prompt.
1. Now for each new token $t$, compute $q_t = x_t W_Q$, $k_t = x_t W_K$ and $v_t = x_t W_V$
2. Append to the KV cache buffers $K \leftarrow [K, k_t]$ and $V \leftarrow [V, v_t]$
3. Use cached $K$ and $V$ from previous tokens.

## References

- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)