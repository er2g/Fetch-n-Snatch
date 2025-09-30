# AnkaDocFlow

Google Drive klasörlerini indirip GPU hızlandırmalı OCR ile metne dönüştüren ve Vertex AI Gemini modelleriyle analiz eden uçtan uca Türkçe belge işleme hattı.

---

## 🚀 Özellikler
- **Drive İndirme** – Service account ile Google Drive klasörlerini yerel diske alır.
- **GPU / CPU OCR** – PDF, PPTX ve görüntüleri EasyOCR + PyTorch ile Türkçe metne dönüştürür.
- **Gemini Analizi** – OCR çıktılarındaki `.txt` dosyalarını Vertex AI (Gemini) modelleriyle parça parça analiz eder, JSON + özet rapor üretir.
- **Web UI** – FastAPI + tek sayfalık arayüzle tüm adımları tarayıcıdan yönet, logları canlı izle.
- **CLI araçları** – Her adım bağımsız Python betikleriyle komut satırından da tetiklenebilir.

---

## 📂 Dizin yapısı
```
├─ drive_fetch.py             # Drive klasörü indirici
├─ gpu_turkish_ocr.py         # OCR boru hattı
├─ analyze_ocr_outputs.py     # Vertex AI Gemini analiz aracı (31.py mantığı)
├─ web_app/                   # FastAPI backend + frontend SPA
└─ sample_run/                # Örnek OCR çıktıları
```

---

## 🔧 Kurulum
1. **Python 3.9+** gereklidir.
2. İsteğe bağlı sanal ortam oluşturun.
3. Gerekli paketleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
4. OCR için **Poppler** kurup `pdftoppm` aracını PATH'e ekleyin.
5. (Opsiyonel) PPTX → PDF dönüşümü için `pip install aspose.slides` (lisanssız kullanım filigran ekleyebilir).
6. Vertex AI tarafında `google-cloud-aiplatform` kullanabilmek için proje ve servis hesabına uygun yetkileri (ör. `Vertex AI User`) atayın ve Gemini model erişimini açın.

> **Not:** GPU kullanmak için CUDA destekli PyTorch kurulumunu kendi donanımınıza uygun komutla yükleyin. (`https://pytorch.org/get-started/locally/`)

---

## ⚙️ Komut satırı kullanımı

### 1. Drive klasörü indir
```bash
python drive_fetch.py <DRIVE_FOLDER_ID> <HEDEF_KLASOR> \
  --service-account /path/drive_sa.json --verbose
```
`--overwrite` mevcut dosyaları yeniden indirir.

### 2. OCR çalıştır
```bash
python gpu_turkish_ocr.py <KAYNAK_KLASOR> \
  --output <CIKTI_KLASORU> --device auto --dpi 220 --verbose
```
- `--device`: `auto | cuda | cpu`
- `--force`: var olan `.txt` dosyalarını yeniden üretir.

### 3. Gemini analizi (Vertex AI)
```bash
python analyze_ocr_outputs.py <OCR_CIKTILARI> \
  --prompt "Belgedeki tıbbi terimleri listele" \
  --service-account /path/gemini_sa.json \
  --model gemini-1.5-flash-002 \
  --region us-central1 --verbose
```
- Çıktılar `<OCR_CIKTILARI>/analysis_outputs` altında `.analysis.json`, `.analysis.txt` ve `combined_report.md` olarak oluşturulur.
- `--chunk-size` ve `--chunk-overlap` değerleri ile metin parçalama ayarlanabilir (varsayılan 6000/300).

---

## 🌐 Web arayüzü
FastAPI sunucusunu başlatın:
```bash
uvicorn web_app.backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Tarayıcıdan `http://localhost:8000` adresine gidin. Panelden sırayla Drive → OCR → Gemini akışını başlatabilir, job loglarını canlı izleyebilir ve gerekirse iptal edebilirsiniz.

WSL üzerinden Windows tarayıcısına servis vermek için örnek script:
```bash
./start_server_wsl.sh 8000
```

---

## 📦 Dağıtıma hazır paketler
- `requirements.txt` – çekirdek bağımlılıklar (EasyOCR, FastAPI, Vertex AI SDK, vb.)
- `.gitignore` – Python, FastAPI ve frontend çıktıları için önerilen filtreler

---

## 📝 Lisans
Projede üçüncü parti kütüphanelerin kendi lisansları geçerlidir. Service account ve Vertex AI/Gemini kullanımında Google Cloud koşullarına uyduğunuzdan emin olun.

---

## 📛 Ad Önerisi
Bu depoyu **“AnkaDocFlow”** adıyla yayınlayabilirsiniz. Anka (Phoenix) hem Türkçe çağrışım yapar hem de dökümanları “yeniden doğuran” boru hattına uygun bir metafor sunar.

> Önerilen açıklama: “AnkaDocFlow – Drive → OCR → Gemini belgeler için uçtan uca Türkçe analiz hattı”.

Mutlu geliştirmeler! 🪶
