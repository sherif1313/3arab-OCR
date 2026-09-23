import os
import tempfile
import fitz
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sherif1313/3arab-OCR-v1"
PDF_PATH = "1.pdf"
OUTPUT_DIR = "./3/output"
DPI = 200
PROMPT = "<image>document parsing."

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, trust_remote_code=True
)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.bfloat16,
).eval().cuda()

print("Model loaded.")


def pdf_to_images(pdf_path, dpi=DPI):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    doc = fitz.open(pdf_path)
    temp_dir = tempfile.mkdtemp(prefix="ocr_")
    scale = dpi / 72
    matrix = fitz.Matrix(scale, scale)
    images = []

    for i, page in enumerate(doc, 1):
        path = os.path.join(temp_dir, f"page_{i:04d}.png")
        page.get_pixmap(matrix=matrix, alpha=False).save(path)
        images.append(path)
        print(f"Converted {i}/{len(doc)}")

    doc.close()
    return images


def ocr_pages(images):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, image in enumerate(images, 1):
        print(f"OCR page {i}/{len(images)}")

        try:
            model.infer(
                tokenizer,
                prompt=PROMPT,
                image_file=image,
                output_path=OUTPUT_DIR,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                max_length=8192,
                no_repeat_ngram_size=35,
                ngram_window=1024,
                save_results=True,
            )
        except Exception as e:
            print(f"Error on page {i}: {e}")


images = pdf_to_images(PDF_PATH)
ocr_pages(images)

print(f"Done. Results: {OUTPUT_DIR}")

