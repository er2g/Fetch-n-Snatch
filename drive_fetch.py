"""Google Drive klasor indirme araci"""
from __future__ import annotations

import argparse
import io
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

GOOGLE_FOLDER_MIME = "application/vnd.google-apps.folder"
GOOGLE_EXPORTS: Dict[str, Tuple[str, str]] = {
    "application/vnd.google-apps.document": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.presentation": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.spreadsheet": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.drawing": ("application/pdf", ".pdf"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Service account ile Google Drive klasorunu tum iceriği ile indirir."
    )
    parser.add_argument("folder_id", help="Indirilecek Drive klasor ID'si")
    parser.add_argument("destination", help="Dosyalarin kaydedilecegi yerel klasor")
    parser.add_argument(
        "--service-account",
        required=True,
        help="Yetkili service account JSON dosyasi",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Var olan dosyalari yeniden indirir",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Detayli loglar",
    )
    return parser.parse_args()


def sanitize_filename(name: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\r\n]+', "_", name)
    sanitized = sanitized.strip()
    return sanitized or "unnamed"


def build_drive_service(service_account_path: Path):
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive indirimi icin google-api-python-client ve google-auth paketleri gerekli: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        ) from exc

    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = service_account.Credentials.from_service_account_file(str(service_account_path), scopes=scopes)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _download_request(request, target_path: Path, downloader_cls) -> None:
    from googleapiclient.http import MediaIoBaseDownload

    target_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    downloader = downloader_cls(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    target_path.write_bytes(buffer.getvalue())


def download_drive_folder(service_account_path: Path, folder_id: str, destination: Path, overwrite: bool) -> None:
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaIoBaseDownload

    service = build_drive_service(service_account_path)
    destination.mkdir(parents=True, exist_ok=True)
    logging.info("Drive klasoru indiriliyor: %s -> %s", folder_id, destination)

    queue: list[tuple[str, Path]] = [(folder_id, destination)]
    visited: set[str] = set()

    while queue:
        current_id, current_dest = queue.pop()
        if current_id in visited:
            continue
        visited.add(current_id)
        current_dest.mkdir(parents=True, exist_ok=True)

        page_token = None
        while True:
            response = service.files().list(
                q=f"'{current_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=page_token,
                pageSize=1000,
            ).execute()

            for item in response.get("files", []):
                file_id = item["id"]
                name = sanitize_filename(item.get("name", "isimsiz"))
                mime_type = item.get("mimeType", "")
                target_path = current_dest / name

                if mime_type == GOOGLE_FOLDER_MIME:
                    queue.append((file_id, target_path))
                    continue

                if not overwrite and target_path.exists():
                    logging.info("Zaten var, atlaniyor: %s", target_path)
                    continue

                try:
                    if mime_type.startswith("application/vnd.google-apps"):
                        export = GOOGLE_EXPORTS.get(mime_type)
                        if not export:
                            logging.warning("Desteklenmeyen Google dosya tipi atlandi: %s (%s)", name, mime_type)
                            continue
                        export_mime, suffix = export
                        if target_path.suffix.lower() != suffix:
                            target_path = target_path.with_suffix(suffix)
                        logging.info("Drive dosyasi export ediliyor: %s -> %s", name, target_path)
                        request = service.files().export_media(fileId=file_id, mimeType=export_mime)
                        _download_request(request, target_path, MediaIoBaseDownload)
                    else:
                        logging.info("Drive dosyasi indiriliyor: %s -> %s", name, target_path)
                        request = service.files().get_media(fileId=file_id)
                        _download_request(request, target_path, MediaIoBaseDownload)
                except HttpError as exc:
                    logging.error("Dosya indirilemedi (%s): %s", target_path, exc)

            page_token = response.get("nextPageToken")
            if not page_token:
                break


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    destination = Path(args.destination).expanduser().resolve()
    service_account_path = Path(args.service_account).expanduser().resolve()

    if not service_account_path.exists():
        logging.error("Service account dosyasi bulunamadi: %s", service_account_path)
        return 1

    try:
        download_drive_folder(service_account_path, args.folder_id, destination, args.overwrite)
    except Exception as exc:  # noqa: BLE001
        logging.error("Drive klasoru indirilemedi: %s", exc)
        return 1

    logging.info("Indirme tamamlandi: %s", destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
