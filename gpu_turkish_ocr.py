"""GPU tabanli Turkce OCR araci"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable

try:
    import easyocr
except ImportError as exc:
    print("easyocr kutuphanesi eksik: pip install easyocr", file=sys.stderr)
    raise

SUPPORTED_IMAGE_EXT = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}
SUPPORTED_PDF_EXT = {".pdf"}
SUPPORTED_PPTX_EXT = {".pptx"}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXT | SUPPORTED_PDF_EXT | SUPPORTED_PPTX_EXT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Secilen klasor ve alt klasorlerindeki pdf, pptx ve gorsel dokumanlari OCR ile tarar."
    )
    parser.add_argument("source", help="Taranacak klasorun yolu")
    parser.add_argument(
        "-o",
        "--output",
        help="OCR sonuc txt dosyalarinin kaydedilecegi klasor. Varsayilan: kaynak klasor icinde ocr_outputs.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="PDF sayfalarinin gorsel donusumunde kullanilacak DPI degeri (varsayilan: 220).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Kullanilacak islemci: auto -> GPU varsa kullanir, yoksa CPU; cuda -> GPU zorunlu; cpu -> sadece CPU.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Varolan txt ciktilarini yeniden olusturur.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=0,
        help="Belirtilen uzunlugun altindaki ciktilari disari yazma (varsayilan: 0).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Detayli loglari etkinlestirir.",
    )
    return parser.parse_args()


def select_device(choice: str) -> bool:
    if choice == "cpu":
        return False
    try:
        import torch
    except ImportError:
        if choice == "cuda":
            raise RuntimeError("PyTorch bulunamadigi icin GPU kullanilamiyor. 'pip install torch' komutunu uygulayin")
        return False
    gpu_available = torch.cuda.is_available()
    if choice == "cuda" and not gpu_available:
        raise RuntimeError("GPU algilanamadi. NVIDIA suruculeri ve CUDA kurulumunu kontrol edin.")
    if choice in {"cuda", "auto"} and gpu_available:
        return True
    return False


def build_reader(use_gpu: bool) -> easyocr.Reader:
    logging.info("EasyOCR yukleniyor (GPU=%s)...", use_gpu)
    return easyocr.Reader(["tr"], gpu=use_gpu)


def extract_from_image(reader: easyocr.Reader, image_path: Path) -> str:
    logging.debug("Gorsel OCR: %s", image_path)
    results = reader.readtext(str(image_path), detail=0, paragraph=True)
    lines = [line.strip() for line in results if line and line.strip()]
    return "\n".join(lines)


def extract_from_pdf(reader: easyocr.Reader, pdf_path: Path, dpi: int) -> str:
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise RuntimeError("pdf2image kutuphanesi eksik: pip install pdf2image") from exc
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy kutuphanesi eksik: pip install numpy") from exc

    logging.debug("PDF OCR: %s", pdf_path)
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    if not pages:
        return ""

    chunks: list[str] = []
    for index, page in enumerate(pages, start=1):
        logging.debug("PDF sayfa %s", index)
        page_array = np.asarray(page)
        results = reader.readtext(page_array, detail=0, paragraph=True)
        page_lines = [line.strip() for line in results if line and line.strip()]
        if page_lines:
            chunks.append(f"--- Sayfa {index} ---")
            chunks.append("\n".join(page_lines))
    return "\n\n".join(chunks)


def convert_pptx_to_pdf(pptx_path: Path, temp_dir: Path) -> Path:
    try:
        import aspose.slides as slides
        from aspose.slides.export import SaveFormat
    except ImportError as exc:
        raise RuntimeError(
            "pptx dosyalarini PDF'e cevirirken aspose.slides paketi gerekli: pip install aspose.slides"
        ) from exc

    pdf_path = temp_dir / (pptx_path.stem + ".pdf")
    with slides.Presentation(str(pptx_path)) as presentation:
        presentation.save(str(pdf_path), SaveFormat.PDF)
    return pdf_path


def find_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def relative_output_path(file_path: Path, source_root: Path, output_root: Path) -> Path:
    rel = file_path.relative_to(source_root)
    return output_root.joinpath(rel).with_suffix(".txt")


def ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    source_root = Path(args.source).expanduser().resolve()
    if not source_root.exists() or not source_root.is_dir():
        logging.error("Kaynak klasor bulunamadi: %s", source_root)
        return 1

    output_root = (
        Path(args.output).expanduser().resolve()
        if args.output
        else source_root.joinpath("ocr_outputs")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        use_gpu = select_device(args.device)
    except RuntimeError as exc:
        logging.error(str(exc))
        return 1

    try:
        reader = build_reader(use_gpu)
    except Exception as exc:
        logging.error("EasyOCR baslatilamadi: %s", exc)
        return 1

    files = list(find_files(source_root))
    if not files:
        logging.warning("Uygun dokuman bulunamadi.")
        return 0

    logging.info("%s dosya isleniyor...", len(files))
    skipped = 0
    processed = 0

    with tempfile.TemporaryDirectory(prefix="pptx2pdf_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        for file_path in files:
            try:
                output_path = relative_output_path(file_path, source_root, output_root)
            except ValueError:
                logging.warning("Dosya kaynak klasorden ayristirilamadi, atlandi: %s", file_path)
                skipped += 1
                continue

            if output_path.exists() and not args.force:
                logging.debug("Zaten var, atlaniyor: %s", output_path)
                skipped += 1
                continue

            logging.info("OCR: %s", file_path)
            suffix = file_path.suffix.lower()

            try:
                if suffix in SUPPORTED_PDF_EXT:
                    text = extract_from_pdf(reader, file_path, args.dpi)
                elif suffix in SUPPORTED_PPTX_EXT:
                    pdf_path = convert_pptx_to_pdf(file_path, tmp_dir_path)
                    text = extract_from_pdf(reader, pdf_path, args.dpi)
                else:
                    text = extract_from_image(reader, file_path)
            except Exception as exc:
                logging.error("Islenemedi (%s): %s", file_path, exc)
                skipped += 1
                continue

            if len(text.strip()) < args.min_length:
                logging.info(
                    "Cikti cok kisa oldugu icin yazilmiyor (%s karakter): %s",
                    len(text.strip()),
                    file_path,
                )
                skipped += 1
                continue

            ensure_directory(output_path)
            output_path.write_text(text, encoding="utf-8")
            processed += 1

    logging.info("Islem tamamlandi: %s dosya yazildi, %s dosya atlandi.", processed, skipped)
    logging.info("Ciktilar: %s", output_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
