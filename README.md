# QLoRA customer-support fine-tuning

Fine-tune a LLaMA-architecture chat model on Northline Outfitters support replies with QLoRA.

QLoRA in this repo means both pieces, in code:

1. **Quantization.** The base model is loaded in 4-bit NF4 with double quantization (`BitsAndBytesConfig`: `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_use_double_quant=True`, compute dtype float16).
2. **LoRA.** PEFT trains adapters on `q_proj`, `k_proj`, `v_proj`, and `o_proj` (rank 8, alpha 16, dropout 0.05). The base weights stay frozen.

Loss is masked to the assistant reply. Prompt tokens are labeled `-100`.

This is separate from [PEFT_Training_LORA](https://github.com/aravindraju98/PEFT_Training_LORA), which is FP16 LoRA on three example turns and does not load a 4-bit model.

## Data

`data/support.jsonl` has 28 customer/agent turns for the fictional retailer Northline. Policies match the companion chatbot repo. Swap in a larger JSONL with the same `instruction` and `response` fields for a real adapter. The default base model is `TinyLlama/TinyLlama-1.1B-Chat-v1.0` because it is public and uses LLaMA attention names. Pass `--model` for a larger checkpoint.

## Requirements

Training needs an NVIDIA GPU and CUDA. `train.py` exits if CUDA is missing, so it will not silently train full-precision LoRA instead.

```bash
pip install -r requirements.txt
python train.py --check
python train.py --max-steps 40
```

`--check` prints the quantization and LoRA settings and does not download a model.

On Colab, with a GPU runtime:

```bash
pip install -r requirements.txt
python train.py --max-steps 40 --output-dir outputs/northline-qlora
```

The adapter is saved with `save_pretrained`. Eval loss is written to `outputs/northline-qlora/metrics.json` after a training run.

## Tests that run without a GPU

```bash
python -m unittest discover -s tests -v
```

The data tests check the dialogue file, prompt formatting, and label masking. The config tests check that the bitsandbytes config is 4-bit NF4 with double quantization and that the LoRA config targets the four attention projections.
