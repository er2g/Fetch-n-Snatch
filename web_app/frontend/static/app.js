const jobsListEl = document.getElementById("jobs-list");
const logOutputEl = document.getElementById("log-output");
const activeJobTitleEl = document.getElementById("active-job-title");
const cancelButton = document.getElementById("cancel-job");
const API_BASE = "/api";

let activeJobId = null;
let jobPollInterval = null;
let jobListInterval = null;

const statusClassMap = {
  running: "status-pill status-running",
  completed: "status-pill status-completed",
  failed: "status-pill status-failed",
  cancelled: "status-pill status-cancelled",
};

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || response.statusText);
  }
  return response.json();
}

function serializeForm(form) {
  const data = new FormData(form);
  const payload = {};
  for (const [key, value] of data.entries()) {
    if (value === "" || value == null) continue;
    if (["overwrite", "force", "verbose"].includes(key)) {
      payload[key] = true;
    } else if (["dpi", "min_length", "max_input_chars", "chunk_overlap", "max_output_tokens", "top_k"].includes(key)) {
      payload[key] = Number(value);
    } else if (["temperature", "top_p"].includes(key)) {
      payload[key] = Number(value);
    } else {
      payload[key] = value;
    }
  }
  for (const checkbox of form.querySelectorAll('input[type="checkbox"]')) {
    if (!checkbox.checked) {
      payload[checkbox.name] = false;
    }
  }
  return payload;
}

function notify(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast toast--${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add("show"));
  setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => toast.remove(), 300);
  }, 4200);
}

function createStatusPill(status) {
  const span = document.createElement("span");
  span.className = statusClassMap[status] || "status-pill";
  span.textContent = status.toUpperCase();
  return span;
}

function renderJobsList(jobs) {
  jobsListEl.innerHTML = "";
  if (!jobs.length) {
    jobsListEl.innerHTML = '<p class="empty">Henüz bir görev bulunmuyor.</p>';
    return;
  }

  jobs.forEach((job) => {
    const item = document.createElement("div");
    item.className = "job-item" + (job.job_id === activeJobId ? " active" : "");

    const title = document.createElement("div");
    title.className = "job-item__title";
    title.textContent = `${job.job_type.toUpperCase()} • ${job.job_id.slice(0, 8)}`;

    const meta = document.createElement("div");
    meta.className = "job-item__meta";
    meta.innerHTML = `<span>${new Date(job.created_at).toLocaleString()}</span>`;
    meta.appendChild(createStatusPill(job.status));

    item.appendChild(title);
    item.appendChild(meta);
    item.addEventListener("click", () => selectJob(job.job_id));
    jobsListEl.appendChild(item);
  });
}

async function refreshJobs() {
  try {
    const jobs = await fetchJSON(`${API_BASE}/jobs`);
    renderJobsList(jobs);
    if (activeJobId && !jobs.some((job) => job.job_id === activeJobId)) {
      clearActiveJob();
    }
  } catch (error) {
    console.error("Job listesi alınamadı", error);
  }
}

async function selectJob(jobId) {
  activeJobId = jobId;
  cancelButton.disabled = false;
  await updateActiveJob();
  if (jobPollInterval) clearInterval(jobPollInterval);
  jobPollInterval = setInterval(updateActiveJob, 2000);
  const jobs = await fetchJSON(`${API_BASE}/jobs`);
  renderJobsList(jobs);
}

async function updateActiveJob() {
  if (!activeJobId) return;
  try {
    const job = await fetchJSON(`${API_BASE}/jobs/${activeJobId}`);
    activeJobTitleEl.textContent = `${job.job_type.toUpperCase()} • ${job.job_id.slice(0, 8)} • ${job.status}`;
    logOutputEl.textContent = job.logs.join("\n");
    logOutputEl.scrollTop = logOutputEl.scrollHeight;
    if (["completed", "failed", "cancelled"].includes(job.status)) {
      cancelButton.disabled = true;
      if (jobPollInterval) clearInterval(jobPollInterval);
    }
  } catch (error) {
    console.error("Log alınamadı", error);
  }
}

function clearActiveJob() {
  activeJobId = null;
  activeJobTitleEl.textContent = "Log akışı";
  logOutputEl.textContent = "Henüz seçili bir görev yok.";
  cancelButton.disabled = true;
  if (jobPollInterval) clearInterval(jobPollInterval);
}

async function cancelActiveJob() {
  if (!activeJobId) return;
  try {
    await fetchJSON(`${API_BASE}/jobs/${activeJobId}/cancel`, { method: "POST" });
    notify("Görev iptal edildi", "warn");
    await updateActiveJob();
  } catch (error) {
    notify(`İptal başarısız: ${error.message}`, "error");
  }
}

async function submitWithEndpoint(form, endpoint, buttonLabel) {
  const submitBtn = form.querySelector(".form-submit");
  submitBtn.disabled = true;
  submitBtn.textContent = "Gönderiliyor...";
  try {
    const payload = serializeForm(form);
    const response = await fetchJSON(`${API_BASE}/${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    notify(`${buttonLabel} başladı (job ${response.job_id.slice(0, 8)})`, "success");
    await refreshJobs();
    selectJob(response.job_id);
  } catch (error) {
    console.error(error);
    notify(`Hata: ${error.message}`, "error");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = buttonLabel;
  }
}

function initToasts() {
  const style = document.createElement("style");
  style.textContent = `
    .toast {
      position: fixed;
      right: 24px;
      top: -60px;
      padding: 14px 20px;
      border-radius: 14px;
      backdrop-filter: blur(10px);
      background: rgba(15, 23, 42, 0.85);
      border: 1px solid rgba(148, 163, 184, 0.2);
      color: #f1f5f9;
      font-weight: 500;
      opacity: 0;
      transform: translateY(-10px);
      transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
      z-index: 9999;
      box-shadow: 0 18px 36px rgba(15, 23, 42, 0.35);
    }
    .toast.show {
      top: 24px;
      opacity: 1;
      transform: translateY(0);
    }
    .toast--success { border-color: rgba(34,197,94,0.45); }
    .toast--error { border-color: rgba(248,113,113,0.5); }
    .toast--warn { border-color: rgba(234,179,8,0.45); }
  `;
  document.head.appendChild(style);
}

function init() {
  initToasts();
  refreshJobs();
  jobListInterval = setInterval(refreshJobs, 5000);

  const driveForm = document.getElementById("drive-form");
  driveForm.addEventListener("submit", (event) => {
    event.preventDefault();
    submitWithEndpoint(driveForm, "run-drive", "Drive İndir");
  });

  const ocrForm = document.getElementById("ocr-form");
  ocrForm.addEventListener("submit", (event) => {
    event.preventDefault();
    submitWithEndpoint(ocrForm, "run-ocr", "OCR Başlat");
  });

  const analysisForm = document.getElementById("analysis-form");
  analysisForm.addEventListener("submit", (event) => {
    event.preventDefault();
    submitWithEndpoint(analysisForm, "run-analysis", "Analizi Başlat");
  });

  cancelButton.addEventListener("click", cancelActiveJob);
}

window.addEventListener("DOMContentLoaded", init);
