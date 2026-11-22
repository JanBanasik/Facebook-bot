# Chat-Style Mimicry with QLoRA

A reproducible experiment and notebook pipeline for testing whether **QLoRA fine-tuning** helps a small instruction-tuned LLM (`Qwen/Qwen2.5-1.5B-Instruct`) mimic your personal messaging style better than **zero-shot prompting** or **few-shot prompting** on held-out chat history.

Everything — data preprocessing, PII scrubbing, training, and inference — runs **100% locally**. No private messages are ever sent to external APIs.

---

## Methodology

The notebook ([`chat_style_lora.ipynb`](chat_style_lora.ipynb)) compares three variants of the exact same base model on identical held-out test turns:

1. **Base** — system style instruction only.
2. **Few-shot** — `N` real examples sampled exclusively from the `train` split.
3. **QLoRA** — 4-bit NF4 LoRA adapter fine-tuned on your replies.

### Key Design Choices

- **Local PII Scrubbing:** Replaces URLs, bare domains, emails, phone numbers, `@handles`, participant names (`[ME]`, `[FRIEND]`), and configurable name patterns with placeholders before any split or model input, verified by assertions.
- **Strict Chronological Split (`80% / 10% / 10%`):** Splits history by time (`train` → `validation` → `test`) rather than random shuffling, preventing future-to-past leakage.
- **Multi-Message Turn Merging:** Consecutive messages from the same sender within a configurable gap (default `180 s`) are merged with `\n` into a single utterance, preserving natural multi-bubble texting rhythm.
- **Target-Only Loss Masking (`-100`):** Every prompt token — including earlier `assistant` turns inside the conversation context — is masked with `-100`. Cross-entropy loss is computed **only** on the target reply and `<EOS>`.
- **8 GB VRAM QLoRA Recipe:** Uses 4-bit NF4 quantization (`bfloat16` compute), LoRA `r=16, alpha=32` on attention + MLP projections, gradient checkpointing (`use_reentrant=False`), `adamw_8bit`, and automatic checkpoint resuming.
- **Qualitative + Diagnostic Evaluation:** Side-by-side HTML comparison on held-out turns, a manual 1–5 rubric (*Sounds like me* & *Fits conversation*), and length/line-count diagnostics (explicitly treated as diagnostics, not proof of style).

---

## Repository Structure

```text
.
├── chat_style_lora.ipynb   # Generalized, privacy-safe experiment notebook (English)
├── pyproject.toml          # Python dependencies (uv / pip)
├── uv.lock                 # Locked dependency versions
├── LICENSE                 # MIT License
└── README.md
```

> Private files such as `data/`, `*.json`, `*.zip`, `artifacts/`, and any personal working notebooks are excluded via `.gitignore`.

---

## Quickstart

### 1. Environment Setup

Requires **Python 3.12+** and an NVIDIA GPU with **8 GB+ VRAM** (tested on RTX 4060 Laptop 8 GB).

Using [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

Or with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Prepare Your Data

Place a Messenger-style JSON export (or `.zip` archive containing it) at `data/messages.json` (or `data/messages.zip`), with the following minimal schema:

```json
{
  "messages": [
    {
      "senderName": "Speaker A",
      "timestamp": 1712066000000,
      "text": "hey, are we still on for today?",
      "isUnsent": false
    }
  ]
}
```

- `timestamp` is in **Unix milliseconds** (or adjust `fmt_ts` / `gap_seconds` if your export uses seconds).
- If `DATA_PATH` does not exist, the notebook raises `FileNotFoundError`.

### 3. Configure & Run

Open [`chat_style_lora.ipynb`](chat_style_lora.ipynb) and update the top configuration cell:

- `ME` — your exact `senderName` in the JSON export
- `FRIEND` — the other participant's `senderName`
- `DATA_PATH` — path to your `.json` or `.zip` export
- `SYSTEM_PROMPT` — adjust the target language note if your chats are not in English (e.g., Polish)

Then run the notebook sequentially:
1. Run preprocessing, assertions, baseline & few-shot generation, and the **2-step smoke test**.
2. Optionally run the **pilot training** (`800` examples, 1 epoch).
3. Set `RUN_FULL_TRAINING = True` to train on the full `train` split (supports automatic checkpoint resuming via `resume_from_checkpoint="auto"`).
4. Inspect the side-by-side comparison table and test custom prompts in the **Interactive Playground** cell at the bottom.

---

## Privacy & Responsible Use

- Do **not** commit raw chat exports, trained LoRA checkpoints (`artifacts/`), or executed notebooks containing personal outputs to a public repository.
- Fine-tuned adapters memorize stylistic patterns and occasional phrasing from training data; always verify outputs before sharing.

---

## License

Released under the [MIT License](LICENSE).
