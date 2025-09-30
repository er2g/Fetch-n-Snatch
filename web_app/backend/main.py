"""FastAPI backend for Drive fetch, OCR and Gemini analysis"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVE_SCRIPT = REPO_ROOT / "drive_fetch.py"
OCR_SCRIPT = REPO_ROOT / "gpu_turkish_ocr.py"
ANALYZE_SCRIPT = REPO_ROOT / "analyze_ocr_outputs.py"
FRONTEND_DIR = REPO_ROOT / "web_app" / "frontend"

for script in (DRIVE_SCRIPT, OCR_SCRIPT, ANALYZE_SCRIPT):
    if not script.exists():
        raise RuntimeError(f"Gerekli betik bulunamadi: {script}")

DEFAULT_MODELS = [
    "gemini-1.5-flash-002",
    "gemini-1.5-pro-002",
    "gemini-1.0-pro",
    "gemini-pro",
]

app = FastAPI(title="GPU Turkish OCR + Gemini Web UI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class Job:
    job_id: str
    command: List[str]
    job_type: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "pending"
    return_code: Optional[int] = None
    logs: List[str] = field(default_factory=list)
    _process: Optional[subprocess.Popen] = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def start(self) -> None:
        threading.Thread(target=self._run, name=f"job-{self.job_id}", daemon=True).start()

    def _append_log(self, line: str) -> None:
        with self._lock:
            self.logs.append(line)
            if len(self.logs) > 2000:
                self.logs = self.logs[-2000:]

    def _run(self) -> None:
        self.status = "running"
        self._append_log("$ " + " ".join(self.command))
        try:
            self._process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:  # noqa: BLE001
            self.status = "failed"
            self.return_code = -1
            self._append_log(f"Komut baslatilamadi: {exc}")
            return

        assert self._process.stdout is not None
        with self._process.stdout:
            for line in self._process.stdout:
                self._append_log(line.rstrip())
        self.return_code = self._process.wait()
        self.status = "completed" if self.return_code == 0 else "failed"
        self._append_log(f"[Islem tamamlandi, kod={self.return_code}]")

    def cancel(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self.status = "cancelled"
            self._append_log("[Islem iptal edildi]")

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            logs_copy = list(self.logs)
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "command": self.command,
            "status": self.status,
            "return_code": self.return_code,
            "created_at": self.created_at.isoformat() + "Z",
            "logs": logs_copy,
        }


jobs: Dict[str, Job] = {}
jobs_lock = threading.Lock()


class DrivePayload(BaseModel):
    folder_id: str
    destination: str
    service_account: str
    overwrite: bool = False
    verbose: bool = True


class OCRPayload(BaseModel):
    source: str
    output: Optional[str] = None
    device: str = Field("auto", pattern="^(auto|cuda|cpu)$")
    dpi: int = Field(220, ge=72)
    min_length: int = Field(0, ge=0)
    force: bool = False
    verbose: bool = True


class AnalysisPayload(BaseModel):
    output_root: str
    prompt: str
    service_account: str
    model: str = Field("gemini-1.5-flash-002")
    region: str = Field("us-central1")
    analysis_dir_name: str = Field("analysis_outputs")
    max_input_chars: int = Field(6000, ge=500)
    chunk_overlap: int = Field(300, ge=0)
    max_output_tokens: int = Field(2048, ge=256)
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(40, ge=1)
    verbose: bool = True


class JobResponse(BaseModel):
    job_id: str
    status: str


class JobDetail(BaseModel):
    job_id: str
    job_type: str
    status: str
    return_code: Optional[int]
    command: List[str]
    created_at: str
    logs: List[str]


class DefaultsResponse(BaseModel):
    models: List[str]


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")


@app.get("/api/defaults", response_model=DefaultsResponse)
async def defaults() -> DefaultsResponse:
    return DefaultsResponse(models=DEFAULT_MODELS)


def _register_job(command: List[str], job_type: str) -> Job:
    job = Job(job_id=str(uuid.uuid4()), command=command, job_type=job_type)
    with jobs_lock:
        jobs[job.job_id] = job
    job.start()
    return job


@app.post("/api/run-drive", response_model=JobResponse)
async def run_drive(payload: DrivePayload) -> JobResponse:
    dest = Path(payload.destination).expanduser().resolve()
    sa_path = Path(payload.service_account).expanduser().resolve()
    if not sa_path.exists():
        raise HTTPException(status_code=400, detail="Service account dosyasi bulunamadi")

    cmd = [
        sys.executable,
        str(DRIVE_SCRIPT),
        payload.folder_id,
        str(dest),
        "--service-account",
        str(sa_path),
    ]
    if payload.overwrite:
        cmd.append("--overwrite")
    if payload.verbose:
        cmd.append("--verbose")

    job = _register_job(cmd, "drive")
    return JobResponse(job_id=job.job_id, status=job.status)


@app.post("/api/run-ocr", response_model=JobResponse)
async def run_ocr(payload: OCRPayload) -> JobResponse:
    source_path = Path(payload.source).expanduser().resolve()
    if not source_path.exists() or not source_path.is_dir():
        raise HTTPException(status_code=400, detail="Kaynak klasor bulunamadi")

    cmd = [sys.executable, str(OCR_SCRIPT), str(source_path)]
    if payload.output:
        cmd.extend(["--output", str(Path(payload.output).expanduser().resolve())])
    cmd.extend(["--device", payload.device])
    cmd.extend(["--dpi", str(payload.dpi)])
    if payload.min_length:
        cmd.extend(["--min-length", str(payload.min_length)])
    if payload.force:
        cmd.append("--force")
    if payload.verbose:
        cmd.append("--verbose")

    job = _register_job(cmd, "ocr")
    return JobResponse(job_id=job.job_id, status=job.status)


@app.post("/api/run-analysis", response_model=JobResponse)
async def run_analysis(payload: AnalysisPayload) -> JobResponse:
    root_path = Path(payload.output_root).expanduser().resolve()
    if not root_path.exists() or not root_path.is_dir():
        raise HTTPException(status_code=400, detail="OCR cikti klasoru bulunamadi")

    sa_path = Path(payload.service_account).expanduser().resolve()
    if not sa_path.exists():
        raise HTTPException(status_code=400, detail="Service account dosyasi bulunamadi")

    cmd = [
        sys.executable,
        str(ANALYZE_SCRIPT),
        str(root_path),
        "--prompt",
        payload.prompt,
        "--service-account",
        str(sa_path),
        "--model",
        payload.model,
        "--region",
        payload.region,
        "--analysis-dir-name",
        payload.analysis_dir_name,
        "--chunk-size",
        str(payload.max_input_chars),
        "--chunk-overlap",
        str(payload.chunk_overlap),
        "--max-output-tokens",
        str(payload.max_output_tokens),
        "--temperature",
        str(payload.temperature),
        "--top-p",
        str(payload.top_p),
        "--top-k",
        str(payload.top_k),
    ]

    if payload.verbose:
        cmd.append("--verbose")

    job = _register_job(cmd, "analysis")
    return JobResponse(job_id=job.job_id, status=job.status)


@app.get("/api/jobs/{job_id}", response_model=JobDetail)
async def job_detail(job_id: str) -> JobDetail:
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job bulunamadi")
    return JobDetail(**job.snapshot())


@app.post("/api/jobs/{job_id}/cancel", response_model=JobDetail)
async def cancel_job(job_id: str) -> JobDetail:
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job bulunamadi")
    job.cancel()
    await asyncio.sleep(0.2)
    return JobDetail(**job.snapshot())


@app.get("/api/jobs", response_model=List[JobDetail])
async def job_list() -> List[JobDetail]:
    with jobs_lock:
        all_jobs = list(jobs.values())
    return [job.snapshot() for job in sorted(all_jobs, key=lambda j: j.created_at, reverse=True)]
