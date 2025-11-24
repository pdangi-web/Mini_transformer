Mini Transformer (toy) - README

- Purpose: small decoder-only transformer implemented from scratch in PyTorch to demonstrate:
  - Manual multi-head self-attention
  - Forward/backward hooks & activation inspection
  - Optional top-k sparse attention
  - Simple training and generation

- Files: (if split into files)
  - model.py -> contains MiniTransformer and components
  - train.py -> data loading and training loop
  - utils.py -> tokenizer and helpers

- How to run (in Colab):
  1. Paste the notebook cells in order.
  2. Change `raw_text` variable to your dataset or upload a .txt file.
  3. Run training cell. For faster experiments, lower `max_steps`.
  4. Use `model.generate` to sample text.

- Extensions you could add:
  - Byte Pair Encoding (BPE) tokenizer (instead of char-level)
  - Larger embed_dim / more layers + gradient checkpointing
  - FSDP / DistributedDataParallel example (Colab single-GPU has limitations)
  - Triton kernel for custom attention (advanced)
  - Compare dense vs sparse attention accuracy & runtime

