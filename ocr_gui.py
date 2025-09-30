"""Tkinter GUI for Drive fetch, OCR, and Gemini analysis"""
from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

REPO_ROOT = Path(__file__).resolve().parent
DRIVE_SCRIPT = REPO_ROOT / "drive_fetch.py"
OCR_SCRIPT = REPO_ROOT / "gpu_turkish_ocr.py"
ANALYZE_SCRIPT = REPO_ROOT / "analyze_ocr_outputs.py"
DEFAULT_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-pro-exp",
]


class CommandThread(threading.Thread):
    def __init__(self, cmd: List[str], log_queue: "queue.Queue[str]", on_finish) -> None:
        super().__init__(daemon=True)
        self.cmd = cmd
        self.log_queue = log_queue
        self.on_finish = on_finish
        self.process: Optional[subprocess.Popen[str]] = None

    def run(self) -> None:
        try:
            self.log_queue.put("$ " + " ".join(self.cmd))
            self.process = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:  # noqa: BLE001
            self.log_queue.put(f"Komut baslatilamadi: {exc}")
            self.on_finish(-1)
            return

        assert self.process.stdout is not None
        with self.process.stdout:
            for line in self.process.stdout:
                self.log_queue.put(line.rstrip("\n"))
        ret = self.process.wait()
        self.on_finish(ret)

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()


class MainWindow(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Drive Fetch + OCR + Gemini Analiz")
        self.geometry("900x680")
        self.minsize(780, 560)

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.current_runner: Optional[CommandThread] = None

        self._build_ui()
        self.after(100, self._poll_log_queue)

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True)

        self.drive_tab = ttk.Frame(notebook, padding=12)
        self.ocr_tab = ttk.Frame(notebook, padding=12)
        self.analysis_tab = ttk.Frame(notebook, padding=12)

        notebook.add(self.drive_tab, text="Drive Indirme")
        notebook.add(self.ocr_tab, text="OCR Taramasi")
        notebook.add(self.analysis_tab, text="Gemini Analizi")

        self._build_drive_tab()
        self._build_ocr_tab()
        self._build_analysis_tab()
        self._build_log_panel()

    def _build_drive_tab(self) -> None:
        frame = self.drive_tab
        row = 0

        self.drive_folder_var = tk.StringVar()
        self.drive_dest_var = tk.StringVar()
        self.drive_sa_var = tk.StringVar()
        self.drive_overwrite_var = tk.BooleanVar(value=False)
        self.drive_verbose_var = tk.BooleanVar(value=True)

        ttk.Label(frame, text="Drive klasor ID").grid(row=row, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.drive_folder_var, width=60).grid(row=row, column=1, sticky="ew", padx=6)
        row += 1

        ttk.Label(frame, text="Kaydedilecek klasor").grid(row=row, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.drive_dest_var, width=60).grid(row=row, column=1, sticky="ew", padx=6)
        ttk.Button(frame, text="Sec", command=lambda: self._pick_dir(self.drive_dest_var)).grid(row=row, column=2)
        row += 1

        ttk.Label(frame, text="Service account JSON").grid(row=row, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.drive_sa_var, width=60).grid(row=row, column=1, sticky="ew", padx=6)
        ttk.Button(frame, text="Sec", command=lambda: self._pick_file(self.drive_sa_var)).grid(row=row, column=2)
        row += 1

        options = ttk.Frame(frame)
        options.grid(row=row, column=0, columnspan=3, sticky="w", pady=6)
        ttk.Checkbutton(options, text="Var olan dosyalari yeniden indir", variable=self.drive_overwrite_var).grid(row=0, column=0, sticky="w", padx=4)
        ttk.Checkbutton(options, text="Verbose", variable=self.drive_verbose_var).grid(row=0, column=1, sticky="w", padx=12)
        row += 1

        self.drive_button = ttk.Button(frame, text="Drive Indirmeyi Baslat", command=self._start_drive)
        self.drive_button.grid(row=row, column=0, columnspan=3, pady=10)

        frame.columnconfigure(1, weight=1)

    def _build_ocr_tab(self) -> None:
        frame = self.ocr_tab
        row = 0

        self.source_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.device_var = tk.StringVar(value="auto")
        self.dpi_var = tk.StringVar(value="220")
        self.force_var = tk.BooleanVar(value=False)
        self.min_length_var = tk.StringVar(value="0")
        self.verbose_var = tk.BooleanVar(value=True)

        ttk.Label(frame, text="Kaynak klasor").grid(row=row, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.source_var, width=60).grid(row=row, column=1, sticky="ew", padx=6)
        ttk.Button(frame, text="Sec", command=lambda: self._pick_dir(self.source_var)).grid(row=row, column=2)
        row += 1

        ttk.Label(frame, text="Cikti klasoru (opsiyonel)").grid(row=row, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.output_var, width=60).grid(row=row, column=1, sticky="ew", padx=6)
        ttk.Button(frame, text="Sec", command=lambda: self._pick_dir(self.output_var)).grid(row=row, column=2)
        row += 1

        ttk.Label(frame, text="GPU ayari").grid(row=row, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=self.device_var, values=["auto", "cuda", "cpu"], width=12).grid(row=row, column=1, sticky="w", padx=6)
        row += 1

        form_line = ttk.Frame(frame)
        form_line.grid(row=row, column=0, columnspan=3, sticky="w", pady=6)
        ttk.Label(form_line, text="PDF DPI").grid(row=0, column=0, sticky="w")
        ttk.Entry(form_line, textvariable=self.dpi_var, width=10).grid(row=0, column=1, sticky="w", padx=(4, 16))
        ttk.Label(form_line, text="Min. metin uzunlugu").grid(row=0, column=2, sticky="w")
        ttk.Entry(form_line, textvariable=self.min_length_var, width=10).grid(row=0, column=3, sticky="w", padx=4)
        ttk.Checkbutton(form_line, text="Var olan ciktilari yenile", variable=self.force_var).grid(row=0, column=4, sticky="w", padx=12)
        ttk.Checkbutton(form_line, text="Verbose", variable=self.verbose_var).grid(row=0, column=5, sticky="w", padx=12)
        row += 1

        self.ocr_button = ttk.Button(frame, text="OCR Taramasini Baslat", command=self._start_ocr)
        self.ocr_button.grid(row=row, column=0, columnspan=3, pady=8)

        frame.columnconfigure(1, weight=1)

    def _build_analysis_tab(self) -> None:
        frame = self.analysis_tab
        row = 0

        self.analysis_root_var = tk.StringVar()
        self.analysis_sa_var = tk.StringVar()
        self.analysis_model_var = tk.StringVar(value=DEFAULT_MODELS[0])
        self.analysis_output_dir_var = tk.StringVar(value="analysis_outputs")
        self.analysis_max_chars_var = tk.StringVar(value="20000")
        self.analysis_max_tokens_var = tk.StringVar(value="1024")
        self.analysis_temp_var = tk.StringVar(value="0.2")
        self.analysis_top_p_var = tk.StringVar(value="0.9")
        self.analysis_top_k_var = tk.StringVar(value="40")

        ttk.Label(frame, text="OCR cikti klasoru").grid(row=row, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.analysis_root_var, width=60).grid(row=row, column=1, sticky="ew", padx=6)
        ttk.Button(frame, text="Sec", command=lambda: self._pick_dir(self.analysis_root_var)).grid(row=row, column=2)
        row += 1

        ttk.Label(frame, text="Service account JSON").grid(row=row, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.analysis_sa_var, width=60).grid(row=row, column=1, sticky="ew", padx=6)
        ttk.Button(frame, text="Sec", command=lambda: self._pick_file(self.analysis_sa_var)).grid(row=row, column=2)
        row += 1

        ttk.Label(frame, text="Model").grid(row=row, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=self.analysis_model_var, values=DEFAULT_MODELS, width=20).grid(row=row, column=1, sticky="w", padx=6)
        row += 1

        ttk.Label(frame, text="Analiz promptu").grid(row=row, column=0, sticky="nw")
        self.prompt_entry = tk.Text(frame, height=6, width=60)
        self.prompt_entry.grid(row=row, column=1, columnspan=2, sticky="ew", padx=6)
        row += 1

        form = ttk.Frame(frame)
        form.grid(row=row, column=0, columnspan=3, sticky="w", pady=6)

        def labelled_entry(c, text, var):
            ttk.Label(form, text=text).grid(row=0, column=c * 2, sticky="w")
            ttk.Entry(form, textvariable=var, width=10).grid(row=0, column=c * 2 + 1, sticky="w", padx=(4, 16))

        labelled_entry(0, "Analiz klasor adi", self.analysis_output_dir_var)
        labelled_entry(1, "Max input", self.analysis_max_chars_var)
        labelled_entry(2, "Max tokens", self.analysis_max_tokens_var)
        labelled_entry(3, "Temp", self.analysis_temp_var)
        labelled_entry(4, "Top-p", self.analysis_top_p_var)
        labelled_entry(5, "Top-k", self.analysis_top_k_var)
        row += 1

        self.analysis_button = ttk.Button(frame, text="Gemini Analizini Baslat", command=self._start_analysis)
        self.analysis_button.grid(row=row, column=0, columnspan=3, pady=8)

        frame.columnconfigure(1, weight=1)

    def _build_log_panel(self) -> None:
        sep = ttk.Separator(self, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, pady=6)

        container = ttk.Frame(self, padding=6)
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(container, text="Cikti / Loglar").pack(anchor="w")
        self.log_text = tk.Text(container, height=10, wrap="word", state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        scrollbar = ttk.Scrollbar(container, command=self.log_text.yview)
        scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        btn_frame = ttk.Frame(container)
        btn_frame.pack(fill=tk.X, pady=4)
        ttk.Button(btn_frame, text="Temizle", command=self._clear_log).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Durdur", command=self._stop_current).pack(side=tk.LEFT, padx=8)

    def _pick_dir(self, var: tk.StringVar) -> None:
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _pick_file(self, var: tk.StringVar) -> None:
        path = filedialog.askopenfilename(filetypes=(("JSON", "*.json"), ("Tum dosyalar", "*.*")))
        if path:
            var.set(path)

    def _append_log(self, line: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, line + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _clear_log(self) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _poll_log_queue(self) -> None:
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._append_log(msg)
        self.after(100, self._poll_log_queue)

    def _stop_current(self) -> None:
        if self.current_runner:
            self.current_runner.stop()
            self.log_queue.put("[Bilgi] durdurma istegi gonderildi.")

    def _start_drive(self) -> None:
        if self.current_runner:
            messagebox.showwarning("Islem devam ediyor", "Once mevcut islemi durdurun.")
            return
        folder = self.drive_folder_var.get().strip()
        dest = self.drive_dest_var.get().strip()
        sa = self.drive_sa_var.get().strip()
        if not folder or not dest or not sa:
            messagebox.showerror("Eksik bilgi", "Drive klasor ID, hedef klasor ve service account belirtin.")
            return
        cmd = [
            sys.executable,
            str(DRIVE_SCRIPT),
            folder,
            dest,
            "--service-account",
            sa,
        ]
        if self.drive_overwrite_var.get():
            cmd.append("--overwrite")
        if self.drive_verbose_var.get():
            cmd.append("--verbose")
        self._run_command(cmd, self.drive_button)

    def _start_ocr(self) -> None:
        if self.current_runner:
            messagebox.showwarning("Islem devam ediyor", "Once mevcut islemi durdurun.")
            return
        source = self.source_var.get().strip()
        if not source:
            messagebox.showerror("Eksik bilgi", "Kaynak klasor secin.")
            return
        cmd = [sys.executable, str(OCR_SCRIPT), source]
        output = self.output_var.get().strip()
        if output:
            cmd.extend(["--output", output])
        device = self.device_var.get().strip()
        if device:
            cmd.extend(["--device", device])
        dpi = self.dpi_var.get().strip()
        if dpi:
            cmd.extend(["--dpi", dpi])
        min_len = self.min_length_var.get().strip()
        if min_len and min_len != "0":
            cmd.extend(["--min-length", min_len])
        if self.force_var.get():
            cmd.append("--force")
        if self.verbose_var.get():
            cmd.append("--verbose")
        self._run_command(cmd, self.ocr_button)

    def _start_analysis(self) -> None:
        if self.current_runner:
            messagebox.showwarning("Islem devam ediyor", "Once mevcut islemi durdurun.")
            return
        root_dir = self.analysis_root_var.get().strip()
        sa_path = self.analysis_sa_var.get().strip()
        prompt = self.prompt_entry.get("1.0", tk.END).strip()
        if not root_dir or not sa_path or not prompt:
            messagebox.showerror("Eksik bilgi", "Klasor, service account ve prompt girin.")
            return
        cmd = [
            sys.executable,
            str(ANALYZE_SCRIPT),
            root_dir,
            "--prompt",
            prompt,
            "--service-account",
            sa_path,
            "--model",
            self.analysis_model_var.get().strip(),
            "--analysis-dir-name",
            self.analysis_output_dir_var.get().strip() or "analysis_outputs",
            "--max-input-chars",
            self.analysis_max_chars_var.get().strip(),
            "--max-output-tokens",
            self.analysis_max_tokens_var.get().strip(),
            "--temperature",
            self.analysis_temp_var.get().strip(),
            "--top-p",
            self.analysis_top_p_var.get().strip(),
            "--top-k",
            self.analysis_top_k_var.get().strip(),
            "--verbose",
        ]
        self._run_command(cmd, self.analysis_button)

    def _run_command(self, cmd: List[str], button: ttk.Button) -> None:
        def on_finish(code: int) -> None:
            if code == 0:
                self.log_queue.put("[Tamamlandi]")
            else:
                self.log_queue.put(f"[Islem bitti: kod {code}]")
            self.current_runner = None
            button.configure(state=tk.NORMAL)

        button.configure(state=tk.DISABLED)
        self.current_runner = CommandThread(cmd, self.log_queue, on_finish)
        self.current_runner.start()


if __name__ == "__main__":
    for script in (DRIVE_SCRIPT, OCR_SCRIPT, ANALYZE_SCRIPT):
        if not script.exists():
            print(f"Gerekli betik bulunamadi: {script}")
            sys.exit(1)
    MainWindow().mainloop()
