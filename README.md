# 🏦 GPT-2 (124M) From Scratch → Continued Pretraining → Banking Chatbot

An end-to-end project: I built a GPT-2 style LLM from scratch in PyTorch, continued pretraining it on my own text data, and then **fine-tuned my own trained weights** (not a ready-made chat model) into a retail-banking customer-support chatbot.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USER/REPO/blob/main/Fine_tuning_GPT2_banking_publish.ipynb)

---

## 📌 Pipeline Overview

```
Stage 1                Stage 2                          Stage 3
Build GPT from    →    Continued pretraining       →    Fine-tune my own weights
scratch (PyTorch)      (GPT-2 124M weights +            on banking Q&A data
                       The Verdict + Gutenberg)         (LoRA, Colab T4) → Gradio demo
                              │
                              ▼
          gpt2_124M_the_verdict_continued_pretraining.pth
```

---

## Stage 1 — Building the LLM from scratch

Following Sebastian Raschka's book *Build a Large Language Model (From Scratch)* (Chapters 1–5), I implemented every component myself in PyTorch:

- GPT-2 BPE tokenization (`tiktoken`) and a sliding-window dataset / dataloader
- Token + positional embeddings
- Causal multi-head self-attention
- LayerNorm, GELU, feed-forward network, residual connections
- The full `GPTModel` (12 layers, 12 heads, 768 embedding dim, 1024 context length ≈ 124M parameters)
- Training loop with train/validation loss, and text generation with temperature and top-k sampling

## Stage 2 — Continued pretraining

- Loaded the **GPT-2 124M** pretrained weights into my from-scratch architecture.
- Continued pretraining it in Google Colab on *The Verdict* plus additional books from Project Gutenberg.
- Saved the result as `gpt2_124M_the_verdict_continued_pretraining.pth` — **this is the base model used in Stage 3.**

## Stage 3 — Fine-tuning my own weights on banking data

**Goal:** turn my own pretrained model into a banking assistant that answers customer questions.

- **Dataset:** [`bitext/Bitext-retail-banking-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset) (`instruction` → `response` pairs), 90/10 train/eval split.
- **Prompt format:**
  ```
  User: <customer question>
  Bot: <answer>
  ```
- **Tokenizer:** the GPT-2 BPE tokenizer (identical to the one used during pretraining).
- **Model:** my own weights, **not** a pre-made chat model.

### Making my weights compatible with Hugging Face

My weights were saved from the book's `GPTModel`, whose parameter names and layout differ from Hugging Face's `GPT2LMHeadModel`. I wrote a loader (`load_my_gpt2`) that converts them exactly:

| From-scratch (`GPTModel`) | Hugging Face (`GPT2LMHeadModel`) |
|---|---|
| `tok_emb`, `pos_emb` | `transformer.wte`, `transformer.wpe` |
| `W_query`, `W_key`, `W_value` (separate) | `attn.c_attn` (concatenated, transposed) |
| `att.out_proj` | `attn.c_proj` |
| `norm1`, `norm2`, `final_norm` (`scale`/`shift`) | `ln_1`, `ln_2`, `ln_f` (`weight`/`bias`) |
| `ff.layers.0`, `ff.layers.2` | `mlp.c_fc`, `mlp.c_proj` |
| `out_head` (untied) | `lm_head` (`tie_word_embeddings=False`) |

Because the activation (tanh-GELU), LayerNorm epsilon and attention scaling match, the converted model is functionally the same network. This lets the standard Hugging Face `Trainer`, `peft`, `generate()` and the Hub tooling work with my from-scratch model.

### Training

- Parameter-efficient fine-tuning with **LoRA** (`peft`): `r=16`, `alpha=32`, `dropout=0.05`, targeting `c_attn`, `c_proj`, `c_fc`.
- Hugging Face `Trainer`, trained on a free **Colab T4 GPU**.
- After training, the LoRA adapters are **merged** into the base weights and the final model is pushed to the Hugging Face Hub.
- A **Gradio** app provides an interactive chat demo.

---

## 🚀 Run it yourself

1. Click the **Open in Colab** badge above and select a **T4 GPU** runtime (`Runtime → Change runtime type`).
2. Run the cells in order. The notebook downloads the pretrained weights from the Hugging Face Hub:
   `https://huggingface.co/USER/gpt2-124m-verdict`
3. Try the chatbot in the last cells (terminal-style chat or the Gradio interface).

> **Note:** if you hit a `torchao` version error from `peft` on Colab, run `!pip install -q -U torchao` (or `!pip uninstall -y torchao`) and restart the session.

## 📊 Results

<!-- Fill in with your own numbers -->
- Final eval loss: `TODO`
- Example conversations / screenshot of the Gradio demo: `TODO`

## ⚠️ Limitations

- This is a **124M-parameter** educational model. Answers can be inaccurate, generic, or repetitive.
- The base model was pretrained on literary text, and the banking behaviour comes only from fine-tuning on a synthetic Q&A dataset.
- **Not for real financial or customer-support use.** Do not rely on it for banking decisions.

## 🛠️ Tech Stack

PyTorch · Hugging Face Transformers · PEFT (LoRA) · Datasets · tiktoken · Gradio · Google Colab (T4)

## 🙏 Credits & Licenses

- Sebastian Raschka — *Build a Large Language Model (From Scratch)*
- OpenAI GPT-2 (MIT License)
- Bitext retail-banking dataset — see its Hugging Face page for the license terms.

## 📬 Contact

Abdelrahman — `LinkedIn / GitHub links here`
