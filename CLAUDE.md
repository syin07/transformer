# About this project
I'm implementing a transformer from scratch to learn, using only PyTorch's
autograd, nn.Module, and nn.Parameter. I'm teaching it to do addition.

# How to help me
- Do not write or edit implementation code unless I explicitly ask.
- When I'm stuck, explain the concept and give hints, not the solution.
- When reviewing my code, point out where the bug is and why, and let me fix it.
- Don't use torch built-ins (nn.Linear, F.softmax, torch.optim, etc.) outside tests.

# Plan
Each step depends only on the ones before it:

1. functional.py: softmax, log-softmax, cross-entropy.
2. layers.py: Linear, Embedding, LayerNorm.
3. optim.py: Adam.
4. data.py
5. attention.py, then block.py, then transformer.py.
6. train.py and evaluate.py.

# Tests 
Functional: compare against torch.softmax, torch.log_softmax, and F.cross_entropy on random inputs with torch.allclose. Include a case with logits around 1000 to check you don't get inf or nan.
Layers: copy your weights into nn.Linear, nn.Embedding, and nn.LayerNorm and assert the outputs match.
Adam: run your optimizer and torch.optim.Adam from identical starting weights on the same small problem for about 10 steps and assert the parameters match after every step. This catches a missing bias correction, the usual Adam bug.

# Structure
addition-transformer/
├── config.py            # Config dataclass (model + training settings)
├── nn_basics/
│   ├── layers.py        # Linear, Embedding, LayerNorm
│   ├── functional.py    # softmax, log_softmax, cross_entropy, activation (ReLU/GELU)
│   └── optim.py         # Adam
├── model/
│   ├── attention.py     # multi-head causal self-attention
│   ├── block.py         # MLP + transformer block
│   └── transformer.py   # embeddings, stack of blocks, output head
├── data.py              # vocab, encode/decode, problem generation, train/test split
├── train.py             # training loop, loss masking, checkpointing
├── evaluate.py          # greedy decoding, exact-match accuracy
└── tests/
    ├── test_layers.py
    ├── test_functional.py
    ├── test_optim.py
    ├── test_attention.py
    ├── test_block.py
    ├── test_transformer.py
    └── test_data.py