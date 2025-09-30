from __future__ import annotations

"""Run Gemini analysis on OCR outputs using Vertex AI, mirroring 31.py logic."""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from google.oauth2 import service_account

try:
    import vertexai
    from vertexai.generative_models import Content, GenerativeModel, GenerationConfig, Part
except ImportError as exc:  # pragma: no cover - dependency missing at runtime
    raise RuntimeError(
        "vertexai paketi bulunamadı. Lütfen `pip install google-cloud-aiplatform` komutunu uygulayın."
    ) from exc


SUPPORTED_TEXT_EXT = {".txt"}
DEFAULT_MODEL = "gemini-1.5-flash-002"
DEFAULT_REGION = "us-central1"
DEFAULT_CHUNK_SIZE = 6000
DEFAULT_CHUNK_OVERLAP = 300
DEFAULT_MAX_OUTPUT_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40

TERM_KEYS = ["terms", "terimler", "keywords", "anahtar_kelimeler"]


@dataclass
class Args:
    output_root: Path
    prompt: str
    service_account: Path
    model: str
    region: str
    analysis_dir_name: str
    chunk_size: int
    chunk_overlap: int
    max_output_tokens: int
    temperature: float
    top_p: float
    top_k: int
    verbose: bool


def parse_args(argv: Optional[List[str]] = None) -> Args:
    parser = argparse.ArgumentParser(
        description="OCR çıktılarındaki .txt dosyalarını Gemini modeliyle analiz eder "
        "(31.py ile aynı mantık)."
    )
    parser.add_argument("output_root", help="OCR çıktı klasörünün yolu.")
    parser.add_argument("--prompt", required=True, help="Analizde kullanılacak talimat metni.")
    parser.add_argument(
        "--service-account",
        required=True,
        help="Vertex AI erişimi için service account JSON dosyası.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Kullanılacak Gemini modeli (varsayılan: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"Vertex AI bölgesi (varsayılan: {DEFAULT_REGION}).",
    )
    parser.add_argument(
        "--analysis-dir-name",
        default="analysis_outputs",
        help="Çıktıların kaydedileceği klasör adı (varsayılan: analysis_outputs).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Gemini çağrısı için parça uzunluğu (varsayılan: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Parçalar arasında korunacak karakter sayısı (varsayılan: {DEFAULT_CHUNK_OVERLAP}).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help=f"Gemini yanıtı için maksimum token sayısı (varsayılan: {DEFAULT_MAX_OUTPUT_TOKENS}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Model temperature değeri (varsayılan: {DEFAULT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help=f"Top-p (varsayılan: {DEFAULT_TOP_P}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Top-k (varsayılan: {DEFAULT_TOP_K}).",
    )
    parser.add_argument("--verbose", action="store_true", help="Detaylı log çıktısı.")

    raw = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if raw.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    output_root = Path(raw.output_root).expanduser().resolve()
    service_account_path = Path(raw.service_account).expanduser().resolve()

    if raw.chunk_size <= 0:
        parser.error("--chunk-size sıfırdan büyük olmalı")
    if raw.chunk_overlap < 0:
        parser.error("--chunk-overlap negatif olamaz")
    if raw.chunk_overlap >= raw.chunk_size:
        parser.error("--chunk-overlap, --chunk-size değerinden küçük olmalı")

    return Args(
        output_root=output_root,
        prompt=raw.prompt,
        service_account=service_account_path,
        model=raw.model,
        region=raw.region,
        analysis_dir_name=raw.analysis_dir_name,
        chunk_size=raw.chunk_size,
        chunk_overlap=raw.chunk_overlap,
        max_output_tokens=max(raw.max_output_tokens, DEFAULT_MAX_OUTPUT_TOKENS),
        temperature=raw.temperature,
        top_p=raw.top_p,
        top_k=raw.top_k,
        verbose=raw.verbose,
    )


def find_text_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_TEXT_EXT:
            yield path


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    index = 0
    length = len(text)
    while index < length:
        end = min(index + chunk_size, length)
        chunks.append(text[index:end])
        if end == length:
            break
        index = max(index + chunk_size - overlap, end)
    return chunks


def try_extract_json(text: str) -> Optional[Dict]:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except Exception:
        pass

    import re

    code_blocks = re.findall(r"```json\s*(\{[\s\S]*?\})\s*```", stripped, re.MULTILINE)
    for candidate in code_blocks:
        try:
            return json.loads(candidate)
        except Exception:
            continue

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None


def aggregate_results(per_chunk_results: List[Dict]) -> Dict:
    agg: Dict[str, object] = {"chunks": per_chunk_results, "summary": {}}
    union: set[str] = set()
    for item in per_chunk_results:
        payload = item.get("parsed_json")
        if isinstance(payload, dict):
            for key in TERM_KEYS:
                value = payload.get(key)
                if isinstance(value, list):
                    union.update({str(term).strip() for term in value if isinstance(term, str)})
    if union:
        agg["summary"]["unique_terms"] = sorted(term for term in union if term)
    return agg


class GeminiRunner:
    def __init__(self, args: Args) -> None:
        self.args = args
        self.credentials = None
        self.project_id: Optional[str] = None
        self.model: Optional[GenerativeModel] = None

    def init(self) -> None:
        try:
            self.credentials = service_account.Credentials.from_service_account_file(
                str(self.args.service_account)
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Service account bulunamadı: {self.args.service_account}") from exc

        if self.project_id is None:
            try:
                data = json.loads(self.args.service_account.read_text(encoding="utf-8"))
                self.project_id = data.get("project_id")
            except Exception as exc:  # noqa: BLE001 - service account parse failure
                raise RuntimeError("Service account JSON okunamadı.") from exc
        if not self.project_id:
            raise RuntimeError("Service account dosyasında project_id alanı bulunamadı.")

        vertexai.init(project=self.project_id, location=self.args.region, credentials=self.credentials)
        system_instruction = (
            "Sen bir metin işleme yardımcısısın. Kullanıcının talimatını aynen uygula. "
            "Mümkünse yanıtını JSON biçiminde döndür (ör: {\"terms\": [...]}) ve anahtar kelimeleri listele. "
            "JSON üretemezsen düz metin dönebilirsin."
        )
        self.model = GenerativeModel(self.args.model, system_instruction=system_instruction)

    def run_chunk(self, prompt: str, chunk_text: str) -> str:
        if self.model is None:
            raise RuntimeError("Gemini modeli başlatılmadı.")
        generation_config = GenerationConfig(
            max_output_tokens=self.args.max_output_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
        )
        contents = [
            Content(
                role="user",
                parts=[
                    Part.from_text(
                        "KULLANICI TALİMATI:\n" + prompt + "\n\n" +
                        "Lütfen mümkünse JSON biçiminde yanıtla (ör: {\"terms\": [...]})"
                    )
                ],
            ),
            Content(role="user", parts=[Part.from_text("METİN PARÇASI:\n" + chunk_text)]),
        ]
        response = self.model.generate_content(contents=contents, generation_config=generation_config)
        return (response.text or "").strip()


def mirror_output_paths(output_root: Path, analysis_dir: Path, file_path: Path) -> Dict[str, Path]:
    rel = file_path.relative_to(output_root)
    rel_dir = rel.parent
    out_dir = analysis_dir.joinpath(rel_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = file_path.stem
    return {
        "json": out_dir / f"{base}.analysis.json",
        "txt": out_dir / f"{base}.analysis.txt",
    }


def process_file(runner: GeminiRunner, file_path: Path, args: Args, total: int, index: int) -> Optional[Dict]:
    logging.info("[%s/%s] İşleniyor: %s", index, total, file_path.relative_to(args.output_root))
    text = read_text(file_path)
    chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)
    logging.info("  ↳ Parça sayısı: %s", len(chunks))

    per_chunk: List[Dict] = []
    for chunk_index, chunk in enumerate(chunks, 1):
        try:
            response_text = runner.run_chunk(args.prompt, chunk)
        except Exception as exc:  # noqa: BLE001
            response_text = f"[MODEL HATA]: {exc}"
            logging.error("    - Chunk %s başarısız: %s", chunk_index, exc)
        parsed = try_extract_json(response_text)
        per_chunk.append(
            {
                "chunk_index": chunk_index,
                "raw_response": response_text,
                "parsed_json": parsed,
            }
        )

    if not per_chunk:
        logging.warning("Boş çıktı nedeniyle atlandı: %s", file_path)
        return None

    aggregated = aggregate_results(per_chunk)

    analysis_dir = args.output_root / args.analysis_dir_name
    paths = mirror_output_paths(args.output_root, analysis_dir, file_path)

    payload = {
        "input_file": str(file_path),
        "relative_path": str(file_path.relative_to(args.output_root)),
        "model": args.model,
        "region": args.region,
        "project_id": runner.project_id,
        "prompt": args.prompt,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "results": aggregated,
    }

    paths["json"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with paths["txt"].open("w", encoding="utf-8") as handle:
        handle.write(f"# Kaynak: {payload['relative_path']}\n")
        handle.write(f"# Model: {args.model} | Bölge: {args.region}\n")
        handle.write(f"# Prompt: {args.prompt}\n")
        summary_terms = aggregated.get("summary", {}).get("unique_terms")
        if summary_terms:
            handle.write("\n## Birleşik Terimler\n")
            for term in summary_terms:
                handle.write(f"- {term}\n")
        handle.write("\n## Parça Bazlı Yanıtlar\n")
        for item in per_chunk:
            handle.write(f"\n--- Chunk {item['chunk_index']} ---\n")
            parsed = item.get("parsed_json")
            if parsed is not None:
                handle.write(json.dumps(parsed, ensure_ascii=False, indent=2))
                handle.write("\n")
            else:
                handle.write(item.get("raw_response", ""))
                handle.write("\n")

    logging.info(
        "  ↳ OK -> %s",
        paths["json"].relative_to(args.output_root / args.analysis_dir_name),
    )
    return payload


def write_combined_report(
    combined_path: Path,
    args: Args,
    results: List[Dict],
) -> None:
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    overall_terms: set[str] = set()
    for payload in results:
        terms = payload.get("results", {}).get("summary", {}).get("unique_terms")
        if isinstance(terms, list):
            overall_terms.update({term for term in terms if isinstance(term, str)})

    with combined_path.open("w", encoding="utf-8") as handle:
        handle.write("# Gemini Analiz Raporu\n")
        handle.write(f"Model: {args.model}\n")
        handle.write(f"Bölge: {args.region}\n")
        handle.write(f"Prompt: {args.prompt}\n")
        handle.write(f"Toplam dosya: {len(results)}\n\n")

        if overall_terms:
            handle.write("## Genel Terimler\n")
            for term in sorted(overall_terms):
                handle.write(f"- {term}\n")
            handle.write("\n")

        for payload in sorted(results, key=lambda item: item["relative_path"]):
            handle.write(f"## {payload['relative_path']}\n")
            terms = payload.get("results", {}).get("summary", {}).get("unique_terms")
            if terms:
                handle.write("- Terimler: " + ", ".join(terms) + "\n")
            else:
                handle.write("- Terim bulunamadı\n")
            handle.write("\n")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if not args.output_root.exists() or not args.output_root.is_dir():
        logging.error("OCR çıktı klasörü bulunamadı: %s", args.output_root)
        return 1
    if not args.service_account.exists():
        logging.error("Service account dosyası bulunamadı: %s", args.service_account)
        return 1

    text_files = sorted(find_text_files(args.output_root))
    if not text_files:
        logging.warning("Analiz edilecek .txt dosyası bulunamadı.")
        return 0

    runner = GeminiRunner(args)
    try:
        runner.init()
    except Exception as exc:  # noqa: BLE001
        logging.error("Gemini istemcisi başlatılamadı: %s", exc)
        return 1

    logging.info(
        "Analiz başlatıldı • dosya sayısı=%s • model=%s • bölge=%s",
        len(text_files),
        args.model,
        args.region,
    )

    processed: List[Dict] = []
    errors = 0

    for index, file_path in enumerate(text_files, 1):
        try:
            payload = process_file(runner, file_path, args, total=len(text_files), index=index)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            logging.error("Dosya işlenemedi (%s): %s", file_path, exc)
            continue
        if payload is None:
            continue
        processed.append(payload)

    if not processed:
        logging.warning("Hiçbir dosya başarıyla işlenemedi.")
        return 1

    combined_path = args.output_root / args.analysis_dir_name / "combined_report.md"
    write_combined_report(combined_path, args, processed)

    logging.info(
        "Analiz tamamlandı • başarı=%s • hata=%s • rapor klasörü=%s",
        len(processed),
        errors,
        args.output_root / args.analysis_dir_name,
    )
    logging.info("Birleşik rapor: %s", combined_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
