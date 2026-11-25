# Vortex Sentinel — Explainable Static Malware Risk Classification

> **Zero-execution static malware triage with per-file SHAP explanations and automated SOC inspection reports.**

---

## Project Metadata

| Field | Details |
| :--- | :--- |
| **Project Title** | Vortex Sentinel — Explainable Static Malware Risk Classification |
| **Hackathon** | AVISHKAR'26 (24-Hour Hackathon), Adichunchanagiri Institute of Technology, Chikkamagaluru |
| **Team Name** | Vortex |
| **Team ID** | `AV26-4C8B10F965` |
| **Domain** | AI in Cyber Security and FinTech |
| **Problem Statement** | 7. Malware File Risk Classification Using Static Features |

---

## Problem & Solution Overview

### The Problem
Security Operations Center (SOC) analysts face an overwhelming daily volume of suspicious executables and binary artifacts. Manual reverse engineering does not scale, while traditional signature-based antivirus engines struggle with polymorphic variants, packed binaries, and obfuscated payloads. At the same time, black-box machine learning detectors often output opaque binary labels or confidence scores without context—leaving analysts unable to verify *why* a file was flagged, increasing alert fatigue and slowing down incident triage.

### The Vortex Sentinel Solution
**Vortex Sentinel** provides fast, safe, **zero-execution static analysis** paired with transparent machine learning explainability:

1. **Safe Static Feature Extraction:** Inspects Portable Executable (PE) metadata and structural properties—including **file size**, **Shannon entropy**, **PE section characteristics**, **API imports**, and **printable strings**—without ever executing the binary.
2. **Risk Scoring via Random Forest:** Classifies static feature vectors using a trained **Random Forest Classifier** to produce a calibrated **0–100 Risk Score** and severity tier.
3. **Per-File SHAP Explanations:** Uses **SHAP (`TreeExplainer`)** to compute exact feature attributions for each individual file, surfacing the top positive and negative risk drivers (e.g., high `.text` section entropy, suspicious Windows API imports, or anomalous section counts).
4. **Automated Analyst Inspection Report & Audit Log:** Generates a structured triage summary and persists inspection records in a local **SQLite** audit log so SOC analysts can review evidence before escalating or whitelisting a file.

---

## System Architecture & Workflow

```text
+-------------------------------------------------------------------+
|                             INPUT                                 |
|        Dataset Feature Record  OR  Safe PE Binary (.exe/.dll)     |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                    STATIC FEATURE EXTRACTOR                       |
|   * Zero-execution parsing via `pefile` & byte-level inspection   |
|   * File size, Shannon entropy, PE sections, API imports, strings |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|             RANDOM FOREST CLASSIFIER + SHAP EXPLAINER             |
|   * Scikit-learn Random Forest -> Probability & Risk Score        |
|   * SHAP TreeExplainer -> Exact per-file feature attributions     |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                        ANALYST OUTPUTS                            |
|   * Calibrated Risk Score (0-100) & Risk Tier (Low/Med/High)      |
|   * Top Feature Reasons (Human-readable SHAP impact drivers)      |
|   * Automated SOC Triage Inspection Report                        |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                 INTERACTIVE DASHBOARD & AUDIT LOG                 |
|   * FastAPI / Streamlit UI for upload, scoring & SHAP visuals     |
|   * SQLite audit log storing timestamps, hashes, scores & reports |
+-------------------------------------------------------------------+
```

---

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Backend / UI** | FastAPI / Streamlit |
| **Machine Learning** | Scikit-learn (`RandomForestClassifier`) |
| **Explainability** | SHAP (`TreeExplainer`) |
| **Data Processing** | Pandas, NumPy |
| **Binary Parsing** | `pefile` |
| **Storage & Audit Trail** | SQLite |

---

## Setup & Run Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd "Vortex Sentinel"
```

### 2. Create and Activate a Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
*(If installing packages manually during development: `pip install fastapi uvicorn streamlit scikit-learn pandas numpy pefile shap`)*

### 4. Run the Application Locally
Depending on the active entry point in your branch:

- **Run the Streamlit Dashboard:**
  ```bash
  streamlit run app.py
  ```
- **Run the FastAPI Service (if using API backend):**
  ```bash
  uvicorn main:app --reload --host 127.0.0.1 --port 8000
  ```

### Team Members *(Update Before Submission)*

| Name | Role | Responsibilities |
| :--- | :--- | :--- |
| `<Member 1 Name>` | `<e.g., Team Lead / ML Engineer>` | `<e.g., Model training, SHAP explainability, evaluation>` |
| `<Member 2 Name>` | `<e.g., Security & Feature Engineer>` | `<e.g., PE feature extractor, entropy & string analysis>` |
| `<Member 3 Name>` | `<e.g., Backend / UI Developer>` | `<e.g., FastAPI/Streamlit dashboard, SQLite audit log>` |
| `<Member 4 Name>` | `<e.g., Documentation & QA>` | `<e.g., Triage reporting, testing, presentation>` |

---

## Model Evaluation & Limitations *(Placeholders to Update After Training)*

### Evaluation Metrics
*(Replace placeholder values below with verified validation/test metrics once model training is complete.)*

| Metric | Score (Placeholder) | Notes |
| :--- | :---: | :--- |
| **Accuracy** | `TBD %` | Overall classification accuracy on held-out test set |
| **Precision** | `TBD %` | Minimizes false alarms on benign software |
| **Recall (Detection Rate)** | `TBD %` | Proportion of malicious samples correctly flagged |
| **F1-Score** | `TBD %` | Harmonic mean of precision and recall |
| **ROC-AUC** | `TBD` | Area under the Receiver Operating Characteristic curve |
| **False Positive Rate (FPR)** | `TBD %` | Benign files mistakenly flagged as high risk |

### Known Limitations
- **Heavily Packed or Encrypted Binaries:** Custom packers, crypters, or commercial obfuscators can mask import tables and distort static section headers, reducing static signal fidelity (though elevated entropy remains a useful indicator).
- **Zero-Day & Living-off-the-Land Evasion:** Novel evasion techniques or malicious payloads embedded inside legitimate signed binaries may evade static heuristics.
- **Triage Aid, Not a Standalone Sandbox:** Vortex Sentinel is designed for rapid pre-screening and analyst decision support. High-risk or inconclusive binaries should still undergo deeper dynamic sandbox execution or manual reverse engineering in an isolated environment.

---

## Disclosures: External Libraries, APIs, Models, Datasets & AI Tools

In accordance with **AVISHKAR'26 Rules (Sections 6.2, 7.2, 10, and 12)**, Team **Vortex** (`AV26-4C8B10F965`) provides the following complete disclosures:

### 1. AI Tools Used
- **Google Antigravity:** Used as an AI-assisted IDE and coding assistant for project scaffolding, code structuring, and refactoring support.
- **Google Gemini:** Used for problem brainstorming, system architecture refinement, and drafting technical documentation.

### 2. Open-Source Libraries & Licenses

| Package | License | Purpose in Vortex Sentinel |
| :--- | :--- | :--- |
| `scikit-learn` | BSD-3-Clause | Training and inference for the Random Forest malware risk classifier and computing evaluation metrics |
| `pandas` | BSD-3-Clause | Tabular feature dataset loading, cleaning, and preprocessing |
| `numpy` | BSD-3-Clause | Numerical array operations, entropy math, and risk score normalization |
| `pefile` | MIT | Safe static parsing of Windows Portable Executable (PE) headers, sections, and import tables |
| `shap` | MIT | `TreeExplainer` computation for per-file feature attribution and explainable triage reasons |
| `fastapi` | MIT | Lightweight REST API layer for static feature extraction and risk scoring endpoints |
| `streamlit` | Apache-2.0 | Interactive SOC analyst dashboard for inspection reports, SHAP charts, and audit logs |

*(Standard Python built-in libraries such as `sqlite3`, `hashlib`, `math`, and `re` are used under the Python Software Foundation License.)*

### 3. External APIs & Pretrained Models
- **External Runtime APIs:** None. All static feature extraction, classification, and SHAP explanation run locally without sending files or telemetry to third-party APIs.
- **Models:** The Random Forest classifier is trained solely on static tabular features using `scikit-learn`; no external opaque pretrained malware weights or third-party cloud threat-intel feeds are queried at runtime.

### 4. Datasets & Data Ethics
- **Authorized & Safe Data Sources Only:** This project uses only publicly available static-feature malware datasets (e.g., academic/benchmark PE feature datasets), synthetically generated test records, and benign system binaries for validation.
- **Zero-Execution Policy:** No live or real-world malware samples are ever executed, detonated, or hosted in this repository. All binary inspection is strictly static and read-only.
- **No Sensitive or Personal Data:** No real financial records, customer information, or personally identifiable information (PII) are collected, stored, or processed by Vortex Sentinel.
