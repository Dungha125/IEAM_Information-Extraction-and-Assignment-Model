# Thư mục `data/` — đây chính là dataset train

Không phải PDF CV thô. Đây là **JSONL chat** để LoRA học sửa field mapping.

| File | Mô tả |
|------|--------|
| `train.jsonl` | Tập huấn luyện (mỗi dòng = 1 mẫu) |
| `val.jsonl` | Tập kiểm tra |
| `manifest.json` | Thống kê số mẫu (sau khi generate) |

## Xem nhanh trên Kaggle / máy local

```bash
# số dòng
wc -l data/train.jsonl data/val.jsonl

# hoặc Python
python -c "print(sum(1 for _ in open('data/train.jsonl',encoding='utf-8')), 'train samples')"

# xem 1 mẫu
python -c "import json; print(json.dumps(json.loads(open('data/train.jsonl',encoding='utf-8').readline()),ensure_ascii=False,indent=2)[:1500])"
```

## Sinh lại / tăng data

```bash
python generate_dataset.py --n 10000 --preview 1
```

Mỗi mẫu gồm: system prompt + user (DRAFT sai + SECTIONS) + assistant (JSON đúng).

## Lưu ý Kaggle

Trong Input sidebar hãy **mở rộng** folder:

`cv-mapper-finetune` → `data` → `train.jsonl`

Nếu không thấy: dataset upload thiếu folder `data/` — upload lại zip hoặc generate trong notebook:

```python
!python generate_dataset.py --n 10000
```
