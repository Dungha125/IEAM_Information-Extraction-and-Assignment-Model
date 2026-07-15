# CV Field Mapper Fine-tune (Qwen2.5-1.5B LoRA)

Standalone repo để train trên **Kaggle A100** (hoặc GPU bất kỳ).

Nhiệm vụ model: sửa placement field CV (wrong draft + sections → JSON đúng slot).

## Files

| File | Role |
|------|------|
| `generate_dataset.py` | Sinh JSONL synthetic VN/EN |
| `prompts.py` | System prompt / schema helpers |
| `train_lora.py` | LoRA SFT (auto bf16 trên A100) |
| `kaggle_train.ipynb` | Notebook chạy 1-click trên Kaggle |
| `data/train.jsonl`, `data/val.jsonl` | Dataset sẵn (~1600 samples) |

## Kaggle (A100)

1. Tạo notebook mới → **Accelerator: GPU T4/P100… hoặc GPU** (A100 nếu có quota).
2. Upload repo này làm Dataset **hoặc** `git clone` trong notebook.
3. Mở `kaggle_train.ipynb` → Run All.
4. Download output folder `lora/` (adapter) về máy.

Hoặc cell nhanh:

```python
%cd /kaggle/working
!git clone https://github.com/<USER>/cv-mapper-finetune.git
%cd cv-mapper-finetune
!pip install -q -r requirements.txt
!python train_lora.py --out /kaggle/working/lora --epochs 2 --batch-size 4 --max-seq-length 4096
```

## Local GPU

```bash
pip install -r requirements.txt
python generate_dataset.py --n 1600   # nếu chưa có data/
python train_lora.py --epochs 2 --batch-size 4
```

## Sau train → Ollama (máy local)

1. Copy `artifacts/lora` về `Interview_processing/Backend/cv_engine/finetune/artifacts/lora`
2. `python export_to_ollama.py` (trong project chính)
3. Convert GGUF + `ollama create cv-mapper:1.5b`

## Defaults A100

- dtype: **bf16**
- batch 4 × grad_accum 4
- LoRA r=16, alpha=32
- max_seq_length 4096
- base: `Qwen/Qwen2.5-1.5B-Instruct`
