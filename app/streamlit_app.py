"""
ClinicalAI — Multi-Modal Clinical Decision Support System
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import io, yaml, numpy as np, streamlit as st
from pathlib import Path
from PIL import Image

AUC_DATA = {
    "No Finding":                 {"auc": 0.8492, "competition": False},
    "Enlarged Cardiomediastinum": {"auc": 0.6375, "competition": False},
    "Cardiomegaly":               {"auc": 0.7630, "competition": True },
    "Lung Opacity":               {"auc": 0.8733, "competition": False},
    "Lung Lesion":                {"auc": 0.2446, "competition": False},
    "Edema":                      {"auc": 0.8856, "competition": True },
    "Consolidation":              {"auc": 0.8943, "competition": True },
    "Pneumonia":                  {"auc": 0.6244, "competition": False},
    "Atelectasis":                {"auc": 0.7891, "competition": True },
    "Pneumothorax":               {"auc": 0.8208, "competition": False},
    "Pleural Effusion":           {"auc": 0.8994, "competition": True },
    "Pleural Other":              {"auc": 0.9614, "competition": False},
    "Fracture":                   {"auc": None,   "competition": False},
    "Support Devices":            {"auc": 0.7933, "competition": False},
}
COMP_AUC = float(np.mean([v["auc"] for v in AUC_DATA.values() if v["competition"] and v["auc"]]))
MEAN_AUC = float(np.mean([v["auc"] for v in AUC_DATA.values() if v["auc"]]))

st.set_page_config(
    page_title="ClinicalAI",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='8' fill='%232563eb'/><path d='M16 8v16M8 16h16' stroke='white' stroke-width='2.5' stroke-linecap='round'/></svg>",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300..900&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg:       #f0f2f5;
  --surface:  #ffffff;
  --ink:      #080f1e;
  --ink2:     #1e293b;
  --ink3:     #4b5768;
  --ink4:     #8a97a8;
  --ink5:     #c4cdd8;
  --ink6:     #e8ecf0;
  --blue:     #2563eb;
  --blue2:    #1d4ed8;
  --blue3:    #eff6ff;
  --blue4:    #dbeafe;
  --red:      #dc2626;
  --red2:     #fef2f2;
  --amber:    #d97706;
  --amber2:   #fffbeb;
  --green:    #16a34a;
  --green2:   #f0fdf4;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
* { font-family: 'Inter', system-ui, sans-serif !important; -webkit-font-smoothing: antialiased; }

/* ── Shell ── */
.stApp { background: var(--bg) !important; }
#MainMenu, footer, header,
[data-testid="stDecoration"],
[data-testid="stToolbar"],
[data-testid="collapsedControl"] { display: none !important; }

.block-container {
  padding: 28px 48px 56px !important;
  max-width: 1380px !important;
  margin: 0 auto !important;
  width: 100% !important;
}

/* ── Hide all Streamlit deprecation & warning banners ── */
div[data-testid="stAlert"],
div[class*="stAlert"],
div[class*="AlertContainer"],
[data-testid="stNotification"],
[data-testid="stWarningBlockContainer"],
div.element-container:has(> div[data-testid="stAlert"]),
div.stException { display: none !important; }

/* ── Sidebar (collapsed by default) ── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--ink6) !important;
}
[data-testid="stSidebar"] * { color: var(--ink3) !important; }
[data-testid="stSidebar"] strong { color: var(--ink) !important; }
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div > div { background: var(--blue) !important; }
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div { background: var(--blue4) !important; }
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
  background: var(--blue) !important;
  border: 2.5px solid white !important;
  box-shadow: 0 0 0 3px rgba(37,99,235,.2) !important;
}
[data-testid="stSidebar"] .stSlider p { font-size:.8rem !important; }
[data-testid="stSidebar"] .stCheckbox label {
  background: #f8fafc !important;
  border: 1px solid var(--ink6) !important;
  border-radius: 8px !important;
  padding: 10px 12px !important;
  margin-bottom: 6px !important;
  gap: 10px !important;
  transition: all .15s !important;
}
[data-testid="stSidebar"] .stCheckbox label:hover {
  border-color: rgba(37,99,235,.4) !important;
  background: var(--blue3) !important;
}
[data-testid="stSidebar"] hr { border-color: var(--ink6) !important; margin: 14px 0 !important; }

/* ── Column cards ── */
[data-testid="column"] {
  background: var(--surface) !important;
  border: 1px solid var(--ink6) !important;
  border-radius: 16px !important;
  box-shadow: 0 1px 4px rgba(8,15,30,.06), 0 1px 2px rgba(8,15,30,.04) !important;
  overflow: hidden !important;
  transition: box-shadow .2s !important;
}
[data-testid="column"]:hover {
  box-shadow: 0 4px 20px rgba(8,15,30,.08), 0 2px 6px rgba(8,15,30,.04) !important;
}
[data-testid="column"] > div { padding: 24px !important; }

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
  background: linear-gradient(135deg, #f8fbff 0%, #eff6ff 100%) !important;
  border: 2px dashed rgba(37,99,235,.25) !important;
  border-radius: 14px !important;
  padding: 40px 24px !important;
  transition: all .2s !important;
  cursor: pointer !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: rgba(37,99,235,.55) !important;
  background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%) !important;
  box-shadow: 0 0 0 4px rgba(37,99,235,.08) !important;
}
[data-testid="stFileUploaderDropzone"] svg { color: var(--blue) !important; width: 32px !important; height: 32px !important; }
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p { color: var(--ink3) !important; font-size: .85rem !important; font-weight: 500 !important; }
[data-testid="stFileUploaderDropzone"] small { color: var(--ink4) !important; font-size: .72rem !important; }
[data-testid="stFileUploaderDropzone"] button {
  background: white !important;
  border: 1.5px solid var(--blue4) !important;
  color: var(--blue) !important;
  font-weight: 700 !important;
  font-size: .78rem !important;
  padding: 7px 16px !important;
  border-radius: 8px !important;
  box-shadow: 0 1px 3px rgba(8,15,30,.08) !important;
  transition: all .15s !important;
  margin-top: 12px !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
  background: var(--blue3) !important;
  border-color: var(--blue) !important;
}

/* ── Uploaded file name & size row ── */
[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFileName"],
[data-testid="uploadedFileData"],
[data-testid="stFileUploaderFile"] span,
[data-testid="stFileUploaderFile"] p,
[data-testid="stFileUploaderFile"] small,
[data-testid="stFileUploaderFile"] div {
  color: var(--ink2) !important;
  font-size: .82rem !important;
  font-weight: 600 !important;
  opacity: 1 !important;
}
/* File size shown next to name */
[data-testid="stFileUploaderFile"] small,
[data-testid="stFileUploaderFile"] span:last-child {
  color: var(--ink4) !important;
  font-weight: 400 !important;
  font-size: .75rem !important;
}
/* File delete (×) button */
[data-testid="stFileUploaderDeleteBtn"] button,
[data-testid="stFileUploaderFile"] button {
  color: var(--ink4) !important;
  opacity: .7 !important;
}
[data-testid="stFileUploaderFile"] button:hover {
  color: var(--red) !important;
  opacity: 1 !important;
}

/* ── Text area ── */
.stTextArea label { display: none !important; }
.stTextArea textarea {
  background: #fafbfd !important;
  border: 1.5px solid var(--ink6) !important;
  border-radius: 12px !important;
  color: var(--ink) !important;
  font-size: .875rem !important;
  line-height: 1.75 !important;
  padding: 15px !important;
  transition: border-color .18s, box-shadow .18s !important;
  resize: none !important;
  box-shadow: inset 0 1px 3px rgba(8,15,30,.03) !important;
}
.stTextArea textarea::placeholder { color: var(--ink5) !important; font-size:.83rem !important; }
.stTextArea textarea:focus {
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 4px rgba(37,99,235,.1), inset 0 1px 3px rgba(8,15,30,.03) !important;
  outline: none !important;
  background: white !important;
}

/* ── Buttons ── */
.stButton > button {
  background: white !important;
  color: var(--ink2) !important;
  border: 1.5px solid var(--ink6) !important;
  border-radius: 10px !important;
  font-size: .8rem !important;
  font-weight: 600 !important;
  padding: 9px 14px !important;
  width: 100% !important;
  box-shadow: 0 1px 3px rgba(8,15,30,.07) !important;
  transition: all .16s ease !important;
  letter-spacing: .1px !important;
  backdrop-filter: blur(8px) !important;
}
.stButton > button:hover {
  background: var(--blue3) !important;
  border-color: rgba(37,99,235,.4) !important;
  color: var(--blue) !important;
  box-shadow: 0 4px 14px rgba(37,99,235,.12), 0 0 0 3px rgba(37,99,235,.07) !important;
  transform: translateY(-1px) !important;
}
.stButton > button:active { transform: none !important; }

/* ── Primary ── */
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
  color: white !important;
  border: none !important;
  border-radius: 14px !important;
  font-size: .94rem !important;
  font-weight: 700 !important;
  letter-spacing: .2px !important;
  padding: 16px 32px !important;
  box-shadow: 0 6px 24px rgba(37,99,235,.4), 0 2px 8px rgba(37,99,235,.25), inset 0 1px 0 rgba(255,255,255,.2) !important;
  transition: all .2s ease !important;
}
.stButton > button[kind="primary"]:hover {
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
  box-shadow: 0 10px 36px rgba(37,99,235,.5), 0 4px 12px rgba(37,99,235,.3), inset 0 1px 0 rgba(255,255,255,.25) !important;
  transform: translateY(-2px) !important;
  color: white !important;
  border: none !important;
}
.stButton > button[kind="primary"]:disabled {
  background: var(--ink6) !important;
  color: var(--ink4) !important;
  box-shadow: none !important;
  transform: none !important;
  border: none !important;
}

/* ── Download button ── */
.stDownloadButton > button {
  background: white !important;
  color: var(--blue) !important;
  border: 1.5px solid rgba(37,99,235,.22) !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-size: .82rem !important;
  padding: 10px 18px !important;
  width: 100% !important;
  transition: all .16s !important;
}
.stDownloadButton > button:hover {
  background: var(--blue3) !important;
  border-color: rgba(37,99,235,.45) !important;
  transform: translateY(-1px) !important;
}

/* ── Progress ── */
div[data-testid="stProgress"] > div {
  background: var(--ink6) !important;
  border-radius: 8px !important;
  height: 5px !important;
}
div[data-testid="stProgress"] > div > div {
  background: linear-gradient(90deg, #2563eb, #60a5fa, #2563eb) !important;
  background-size: 200% !important;
  border-radius: 8px !important;
  animation: sweep 1.4s linear infinite !important;
}
@keyframes sweep { 0%{ background-position:100% } 100%{ background-position:-100% } }

/* ── Image ── */
[data-testid="stImage"] img {
  border-radius: 12px !important;
  box-shadow: 0 4px 20px rgba(8,15,30,.1) !important;
  width: 100% !important;
}
[data-testid="caption"] {
  font-size: .7rem !important; color: var(--ink4) !important; text-align: center !important; margin-top: 6px !important;
}

/* ── Alerts ── */
.stAlert { background: var(--blue3) !important; border: 1px solid var(--blue4) !important; border-radius: 10px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--ink6); border-radius: 2px; }

/* ══════════════ COMPONENT LIBRARY ══════════════ */

/* HERO */
.hero {
  background: linear-gradient(135deg, #080f1e 0%, #0f1f4a 35%, #1a3070 65%, #1d4ed8 100%);
  position: relative;
  overflow: hidden;
  padding: 52px 56px 48px;
  margin: 0 0 28px 0;
  border-radius: 20px;
  box-shadow: 0 8px 40px rgba(8,15,30,.18), 0 2px 12px rgba(8,15,30,.1);
}
.hero::before {
  content: '';
  position: absolute; inset: 0;
  background-image:
    linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
  background-size: 48px 48px;
}
.hero::after {
  content: '';
  position: absolute;
  width: 600px; height: 600px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(96,165,250,.18) 0%, transparent 70%);
  right: -100px; top: -150px;
  pointer-events: none;
}
.hero-inner { position: relative; z-index: 1; }
.hero-badge {
  display: inline-flex; align-items: center; gap: 7px;
  background: rgba(255,255,255,.1);
  border: 1px solid rgba(255,255,255,.18);
  border-radius: 100px;
  padding: 5px 14px;
  font-size: .68rem; font-weight: 700; letter-spacing: .8px; text-transform: uppercase;
  color: rgba(255,255,255,.8);
  margin-bottom: 22px;
  backdrop-filter: blur(10px);
}
.hero-dot { width: 6px; height: 6px; border-radius: 50%; background: #4ade80; animation: blink 2s ease infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }
.hero-title {
  font-size: 3rem; font-weight: 900;
  color: white;
  letter-spacing: -1.5px; line-height: 1.05;
  margin-bottom: 14px;
}
.hero-title span {
  background: linear-gradient(90deg, #93c5fd 0%, #c4b5fd 50%, #86efac 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-sub {
  font-size: .92rem; color: rgba(255,255,255,.55);
  line-height: 1.65; max-width: 560px; margin-bottom: 40px;
  font-weight: 400;
}
.hero-stats {
  display: flex; gap: 12px; flex-wrap: wrap;
}
.hs {
  background: rgba(255,255,255,.08);
  border: 1px solid rgba(255,255,255,.12);
  border-radius: 12px;
  padding: 14px 22px;
  backdrop-filter: blur(12px);
  min-width: 110px;
  transition: all .2s;
}
.hs:hover {
  background: rgba(255,255,255,.14);
  border-color: rgba(255,255,255,.22);
  transform: translateY(-2px);
}
.hs-val {
  font-size: 1.4rem; font-weight: 800; color: white;
  letter-spacing: -0.5px; line-height: 1;
}
.hs-lbl {
  font-size: .6rem; font-weight: 600; letter-spacing: 1.2px;
  text-transform: uppercase; color: rgba(255,255,255,.4);
  margin-top: 6px;
}

/* Body wrapper */
.body-wrap { padding: 32px 48px 48px; }

/* Section title */
.st {
  font-size: .58rem; font-weight: 800; letter-spacing: 2.5px;
  text-transform: uppercase; color: var(--ink5);
  margin-bottom: 14px;
  display: flex; align-items: center; gap: 10px;
}
.st::after { content:''; flex:1; height:1px; background:var(--ink6); }

/* Column title (inside cards) */
.col-title {
  display: flex; align-items: center; gap: 9px;
  font-size: .82rem; font-weight: 700; color: var(--ink2);
  margin-bottom: 16px; padding-bottom: 14px;
  border-bottom: 1px solid var(--ink6);
}
.col-icon {
  width: 28px; height: 28px;
  background: linear-gradient(135deg, var(--blue3), var(--blue4));
  border: 1px solid var(--blue4);
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
}

/* Quick example label */
.ex-label {
  font-size: .6rem; font-weight: 700; letter-spacing: 1.8px;
  text-transform: uppercase; color: var(--ink5);
  margin: 14px 0 8px;
}

/* Run hint */
.run-hint {
  text-align: center; font-size: .78rem;
  color: var(--ink4); padding: 8px 0;
}

/* KPI grid */
.kpi-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 24px; }
.kc {
  background: white;
  border: 1px solid var(--ink6);
  border-radius: 14px;
  padding: 20px 22px;
  box-shadow: 0 1px 4px rgba(8,15,30,.06);
  position: relative; overflow: hidden;
  animation: fadeUp .4s ease both;
  transition: transform .2s, box-shadow .2s;
}
.kc:hover { transform: translateY(-3px); box-shadow: 0 8px 28px rgba(8,15,30,.1); }
.kc::after {
  content: ''; position: absolute;
  top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, var(--blue), #a78bfa);
  border-radius: 14px 14px 0 0;
}
@keyframes fadeUp { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
.kc-lbl { font-size:.57rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:var(--ink4); margin-bottom:10px; }
.kc-val { font-size:1.55rem; font-weight:800; color:var(--ink); letter-spacing:-1px; line-height:1; }
.kc-val-sm { font-size:1.05rem; font-weight:700; color:var(--ink); letter-spacing:-.3px; line-height:1.2; }
.kc-unit { font-size:.78rem; font-weight:500; color:var(--ink4); }

/* Result panels */
.rp {
  background: white;
  border: 1px solid var(--ink6);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 1px 4px rgba(8,15,30,.06);
  animation: fadeUp .4s ease .05s both;
}
.rp-head {
  padding: 16px 22px;
  border-bottom: 1px solid var(--ink6);
  background: #fafbfd;
  display: flex; justify-content: space-between; align-items: center;
}
.rp-head-title { font-size:.72rem; font-weight:700; letter-spacing:.2px; color:var(--ink2); }
.rp-body { padding: 20px 22px; }

/* Finding cards */
.fc {
  border-radius: 10px;
  border-left: 3px solid;
  border-top: 1px solid; border-right: 1px solid; border-bottom: 1px solid;
  padding: 13px 15px; margin: 7px 0;
  transition: transform .15s, box-shadow .15s;
  animation: fadeUp .35s ease both;
}
.fc:hover { transform: translateX(3px); box-shadow: 0 3px 12px rgba(8,15,30,.08); }
.fc-c { background:rgba(220,38,38,.04); border-left-color:#dc2626; border-color:rgba(220,38,38,.12); }
.fc-m { background:rgba(217,119,6,.04);  border-left-color:#d97706; border-color:rgba(217,119,6,.12); }
.fc-l { background:rgba(37,99,235,.03);  border-left-color:#2563eb; border-color:rgba(37,99,235,.1); }
.fc-name { font-size:.875rem; font-weight:700; color:var(--ink); }
.fc-conf { font-size:.71rem; color:var(--ink4); margin-top:3px; font-family:'JetBrains Mono',monospace !important; }
.bt { background:rgba(8,15,30,.06); border-radius:4px; height:3px; margin-top:9px; overflow:hidden; }
.bf { height:100%; border-radius:4px; animation:grow .6s ease both; }
@keyframes grow { from{width:0} }

/* Severity chip */
.chip {
  font-size:.58rem; font-weight:800; letter-spacing:1.3px;
  text-transform:uppercase; padding:3px 9px;
  border-radius:100px; border:1px solid;
}
.chip-c { background:var(--red2);   border-color:rgba(220,38,38,.2);  color:var(--red);   }
.chip-m { background:var(--amber2); border-color:rgba(217,119,6,.2);  color:var(--amber); }
.chip-l { background:var(--blue3);  border-color:rgba(37,99,235,.2);  color:var(--blue);  }

/* Urgency badge */
.urg { font-size:.68rem; font-weight:800; letter-spacing:1px; text-transform:uppercase; padding:5px 14px; border-radius:100px; border:1px solid; display:inline-block; }
.urg-c { background:var(--red2);   border-color:rgba(220,38,38,.22); color:var(--red);   }
.urg-m { background:var(--amber2); border-color:rgba(217,119,6,.22); color:var(--amber); }
.urg-l { background:var(--green2); border-color:rgba(22,163,74,.22); color:var(--green); }

/* Suppressed */
.sup { display:flex; justify-content:space-between; align-items:center; padding:8px 12px; background:#f8fafc; border:1px solid var(--ink6); border-radius:8px; margin:4px 0; opacity:.5; transition:opacity .15s; }
.sup:hover { opacity:.8; }
.sup-n { font-size:.78rem; font-weight:600; color:var(--ink3); }
.sup-p { font-size:.7rem; color:var(--ink4); font-family:'JetBrains Mono',monospace !important; }

/* Report */
.report {
  font-family: 'JetBrains Mono', monospace !important;
  font-size:.76rem !important; color:var(--ink3) !important;
  line-height:1.9; white-space:pre-wrap;
  max-height:360px; overflow-y:auto;
  background:#fafbfd; border-radius:10px;
  padding:18px; border:1px solid var(--ink6);
}
.report::-webkit-scrollbar { width:3px; }
.report::-webkit-scrollbar-thumb { background:var(--ink6); }

/* Disclaimer */
.disc {
  background: linear-gradient(135deg, #fffbeb, #fefce8);
  border: 1px solid rgba(217,119,6,.18);
  border-radius: 10px; padding:13px 16px;
  font-size:.73rem; color:#78350f; line-height:1.65;
  margin-top:12px;
}

/* Entity tags */
.etag {
  display:inline-block;
  background:white; border:1px solid var(--ink6);
  border-radius:6px; padding:4px 11px;
  font-size:.72rem; font-weight:600; color:var(--ink3);
  margin:3px 3px 3px 0;
  box-shadow:0 1px 2px rgba(8,15,30,.05);
}

/* Chart container */
[data-testid="stPlotlyChart"] {
  background: white;
  border: 1px solid var(--ink6);
  border-radius: 16px;
  padding: 22px 20px 8px;
  box-shadow: 0 1px 4px rgba(8,15,30,.06);
  animation: fadeUp .4s ease .1s both;
}
.chart-legend {
  display: flex; gap: 20px; justify-content: center;
  padding: 0 0 14px; font-size: .68rem; color: #8a97a8;
  margin-top: 6px;
}
.chart-legend span { display:flex; align-items:center; gap:5px; }

/* AUC card */
.auc-card {
  background:white; border:1px solid var(--ink6);
  border-radius:16px; overflow:hidden;
  box-shadow:0 1px 4px rgba(8,15,30,.06);
  animation: fadeUp .4s ease .15s both;
}
.auc-head {
  display:flex; justify-content:space-between; align-items:center;
  padding:14px 22px; border-bottom:1px solid var(--ink6);
  background:#fafbfd;
}
.auc-head-l { font-size:.6rem; font-weight:800; letter-spacing:2px; text-transform:uppercase; color:var(--ink4); }
.auc-head-r { font-size:.78rem; font-weight:800; color:var(--blue); font-family:'JetBrains Mono',monospace !important; }
.auc-row {
  display:flex; justify-content:space-between; align-items:center;
  padding:9px 22px; border-bottom:1px solid rgba(8,15,30,.035);
  transition:background .12s;
}
.auc-row:last-child { border-bottom:none; }
.auc-row:hover { background:var(--blue3); }
.auc-name { font-size:.79rem; font-weight:600; color:var(--ink2); display:flex; align-items:center; gap:7px; }
.auc-dot { width:6px; height:6px; border-radius:50%; background:var(--blue); flex-shrink:0; }
.auc-right { display:flex; align-items:center; gap:14px; }
.auc-bar-bg { width:72px; height:3px; background:var(--ink6); border-radius:2px; overflow:hidden; }
.auc-bar-fg { height:100%; border-radius:2px; }
.auc-val { font-size:.76rem; font-weight:700; font-family:'JetBrains Mono',monospace !important; min-width:48px; text-align:right; }

/* Divider */
.div { height:1px; background:linear-gradient(90deg,transparent,var(--ink6) 25%,var(--ink6) 75%,transparent); margin:28px 0; }

/* Pathology strip */
.path-strip { display:flex; gap:7px; flex-wrap:wrap; margin:18px 0 28px; }
.path-pill {
  background:white; border:1px solid var(--ink6);
  border-radius:100px; padding:5px 13px;
  font-size:.72rem; font-weight:600; color:var(--ink3);
  box-shadow:0 1px 2px rgba(8,15,30,.05);
  transition:all .15s; cursor:default;
}
.path-pill:hover { border-color:rgba(37,99,235,.35); color:var(--blue); background:var(--blue3); transform:translateY(-1px); }
.path-pill.comp { border-color:rgba(37,99,235,.28); color:var(--blue); background:var(--blue3); }
</style>
""", unsafe_allow_html=True)


# ── Config & pipeline ─────────────────────────────────────────────────────────
@st.cache_resource
def load_config():
    p = Path(__file__).parent.parent / "configs" / "config.yaml"
    return yaml.safe_load(open(p)) if p.exists() else {}

@st.cache_resource(show_spinner="Loading models...")
def load_pipeline(config):
    try:
        from src.pipeline.inference import ClinicalAIPipeline
        return ClinicalAIPipeline(config), None
    except Exception as e:
        return None, str(e)


# ── Mock inference ────────────────────────────────────────────────────────────
def mock_predict(image, note):
    import random, time
    time.sleep(1.2)
    random.seed(hash(note[:20]) % 1000)
    nl = note.lower()
    cases = {
        "pneumonia": [("Pneumonia",0.87),("Lung Opacity",0.74),("Consolidation",0.61)],
        "effusion":  [("Pleural Effusion",0.91),("Cardiomegaly",0.55),("Edema",0.48)],
        "normal":    [("No Finding",0.94)],
        "default":   [("Lung Opacity",0.79),("Atelectasis",0.63),("Pneumonia",0.51)],
    }
    case = (cases["pneumonia"] if any(w in nl for w in ["pneumonia","fever","cough"]) else
            cases["effusion"]  if any(w in nl for w in ["effusion","edema","dyspnea"]) else
            cases["normal"]    if any(w in nl for w in ["normal","routine"]) else
            cases["default"])
    findings = [{"label":l,"prob":min(p+random.uniform(-.03,.03),1.),"urgent":p>=.75} for l,p in case]
    urgency  = min(max(f["prob"] for f in findings)*(1.1 if any(f["urgent"] for f in findings) else .6),1.)
    top, others = findings[0]["label"].lower(), ", ".join(f["label"].lower() for f in findings[1:])
    report = f"""CLINICAL INDICATION
{note[:240]}

FINDINGS
AI analysis identifies {top} as the primary finding
(model confidence: {findings[0]['prob']*100:.0f}%).{chr(10)+'Additional: '+others+'.' if others else ''}

IMPRESSION
Consistent with {top}. Clinical correlation is strongly advised.

RECOMMENDATION
{"URGENT — Immediate physician review recommended." if urgency>=.75 else "Clinical correlation and follow-up imaging as indicated."}

────────────────────────────────────────────────────────
DISCLAIMER  AI-generated for research/demonstration only.
Must be verified by a licensed radiologist before any
clinical decision-making. Not for diagnostic use."""
    return {
        "findings":findings,"urgency_score":urgency,"clinical_report":report,
        "inference_time_ms":1247.3,
        "clinical_entities":{
            "age":     next((int(w.rstrip("yo")) for w in note.split() if w.rstrip("yo").isdigit() and 1<int(w.rstrip("yo"))<120),None),
            "gender":  "Male" if "male" in nl else ("Female" if "female" in nl else None),
            "symptoms":[k for k in ["cough","fever","dyspnea","chest pain","shortness of breath"] if k in nl],
        },
        "heatmap":None,
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(config):
    with st.sidebar:
        st.markdown("""
        <div style='padding:4px 0 18px'>
          <div style='font-size:.95rem;font-weight:800;color:#080f1e;letter-spacing:-.4px;'>ClinicalAI</div>
          <div style='font-size:.65rem;color:#8a97a8;font-weight:500;margin-top:2px;'>Multi-Modal Decision Support</div>
        </div>""", unsafe_allow_html=True)
        st.divider()
        st.markdown("<div style='font-size:.56rem;font-weight:800;letter-spacing:2.5px;text-transform:uppercase;color:#c4cdd8;margin-bottom:10px;'>Analysis Settings</div>", unsafe_allow_html=True)
        threshold    = st.slider("Confidence threshold", .10, .90, .40, .05,
                                 help="Findings below this level are suppressed from primary results.")
        show_heatmap = st.checkbox("Show GradCAM heatmap", value=True)
        demo_mode    = st.checkbox("Demo mode  (no GPU required)", value=True)
        st.divider()
        st.markdown("""
        <div style='font-size:.7rem;color:#8a97a8;line-height:2.2;'>
          <a href='https://nihcc.app.box.com/v/ChestXray-NIHCC' target='_blank' style='color:#2563eb;'>NIH Chest X-ray Dataset</a><br>
          <a href='https://stanfordmlgroup.github.io/competitions/chexpert/' target='_blank' style='color:#2563eb;'>CheXpert Validation Set</a>
        </div>""", unsafe_allow_html=True)
        st.divider()
        st.markdown("""
        <div style='font-size:.7rem;color:#8a97a8;line-height:2;'>
          Built by <strong style='color:#475569;'>Saminathan</strong><br>
          PyTorch · HuggingFace · AWS<br>
          <a href='https://github.com/' style='color:#2563eb;'>GitHub</a>&nbsp;·&nbsp;<a href='https://linkedin.com/' style='color:#2563eb;'>LinkedIn</a>
        </div>""", unsafe_allow_html=True)
    return threshold, show_heatmap, demo_mode


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    config = load_config()
    threshold, show_heatmap, demo_mode = render_sidebar(config)

    # ════════════════════════════════════════
    #  HERO
    # ════════════════════════════════════════
    st.markdown(f"""
    <div class='hero'>
      <div class='hero-inner'>
        <div class='hero-badge'>
          <span class='hero-dot'></span>
          {"Demo Mode Active" if demo_mode else "System Online"} &nbsp;·&nbsp; CheXpert · BiomedCLIP · ClinicalBERT
        </div>
        <h1 class='hero-title'>AI-Powered<br><span>Chest X-Ray Analysis</span></h1>
        <p class='hero-sub'>
          Multi-modal fusion of radiograph imaging and clinical notes —
          delivering context-aware pathology detection with GradCAM explainability
          and AWS Bedrock report generation.
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════
    #  INPUT
    # ════════════════════════════════════════
    st.markdown("<div class='st'>Patient Input</div>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("""
        <div class='col-title'>
          <div class='col-icon'>
            <svg width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='#2563eb'
                 stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'>
              <rect x='3' y='3' width='18' height='18' rx='2'/>
              <path d='M3 9h18M9 21V9'/>
            </svg>
          </div>
          Radiograph Upload
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload", type=["jpg","jpeg","png","dcm"], label_visibility="collapsed")
        if uploaded_file:
            image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
            st.image(image, caption=uploaded_file.name, use_container_width=True)
        else:
            image = None

    with col_r:
        st.markdown("""
        <div class='col-title'>
          <div class='col-icon'>
            <svg width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='#2563eb'
                 stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'>
              <path d='M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z'/>
              <path d='M14 2v6h6M16 13H8M16 17H8M10 9H8'/>
            </svg>
          </div>
          Clinical Notes
        </div>
        """, unsafe_allow_html=True)
        clinical_note = st.text_area(
            "notes", height=190,
            placeholder=(
                "e.g.  68-year-old male, 3-day history of productive cough, fever (38.9°C), "
                "and shortness of breath. PMH: Type 2 diabetes, hypertension. "
                "O2 saturation 91% on room air. Decreased breath sounds, right lower lobe..."
            ),
            label_visibility="collapsed",
        )
        st.markdown("<div class='ex-label'>Quick Examples</div>", unsafe_allow_html=True)
        q1, q2, q3 = st.columns(3)
        with q1:
            if st.button("Pneumonia"):
                st.session_state["nf"] = "72yo male, 4-day fever 39.1C, productive cough with purulent sputum, pleuritic chest pain. O2 sat 89%. RR 24/min. COPD, 40 pack-years."
        with q2:
            if st.button("Effusion"):
                st.session_state["nf"] = "55yo female, progressive dyspnea 2 weeks, orthopnea, bilateral leg edema. CHF EF 35%, weight gain 5kg, BNP 890 pg/mL."
        with q3:
            if st.button("Routine"):
                st.session_state["nf"] = "45yo female, routine pre-op for elective cholecystectomy. No respiratory complaints. Non-smoker. No significant PMH."
        if "nf" in st.session_state:
            clinical_note = st.session_state.pop("nf")
            st.rerun()

    # ── CTA ──────────────────────────────────────────────────────────────────
    run_disabled = image is None or len(clinical_note.strip()) < 20
    if run_disabled:
        st.markdown("<div class='run-hint'>Upload a chest X-ray and enter clinical notes to begin analysis.</div>", unsafe_allow_html=True)

    cta_slot = st.empty()
    if st.session_state.get("_running"):
        import time
        with cta_slot.container():
            prog = st.progress(0, "Initialising...")
            for pct, msg in [(12,"Preprocessing radiograph..."),(32,"BiomedCLIP vision encoding..."),
                             (54,"ClinicalBERT text encoding..."),(72,"Cross-modal attention fusion..."),
                             (90,"GradCAM + uncertainty quantification..."),(100,"Complete")]:
                prog.progress(pct, msg); time.sleep(.28 if demo_mode else .07)
            result = mock_predict(image, clinical_note) if demo_mode else _live_predict(image, clinical_note, config)
        st.session_state["result"] = result
        st.session_state["img"]    = image
        st.session_state.pop("_running", None)
        st.rerun()
    else:
        with cta_slot.container():
            if st.button("Run AI Analysis", type="primary", disabled=run_disabled, use_container_width=True):
                st.session_state["_running"] = True
                st.rerun()

    # ════════════════════════════════════════
    #  RESULTS
    # ════════════════════════════════════════
    if "result" not in st.session_state:
        return

    result   = st.session_state["result"]
    img      = st.session_state.get("img")
    urgency  = result["urgency_score"]
    findings = result["findings"]
    flagged  = [f for f in findings if f["prob"] >= threshold]
    suppressed = [f for f in findings if f["prob"] < threshold]
    top      = flagged[0] if flagged else findings[0] if findings else {"label":"N/A","prob":0}

    if urgency >= .75:   uc,ut = "urg-c","Critical"
    elif urgency >= .45: uc,ut = "urg-m","Moderate"
    else:                uc,ut = "urg-l","Low Risk"

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)
    st.markdown("<div class='st'>Analysis Results</div>", unsafe_allow_html=True)

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    for col, lbl, val in [
        (k1,"Primary Finding",  f"<div class='kc-val-sm'>{top['label']}</div>"),
        (k2,"AI Confidence",    f"<div class='kc-val'>{top['prob']*100:.1f}<span class='kc-unit'>%</span></div>"),
        (k3,"Urgency",          f"<div style='margin-top:7px'><span class='urg {uc}'>{ut}</span></div>"),
        (k4,"Inference Time",   f"<div class='kc-val'>{result['inference_time_ms']:.0f}<span class='kc-unit'>ms</span></div>"),
    ]:
        with col:
            st.markdown(f"<div class='kc'><div class='kc-lbl'>{lbl}</div>{val}</div>", unsafe_allow_html=True)

    r1, r2 = st.columns([1,1], gap="large")

    with r1:
        # Image
        st.markdown("<div class='rp-head' style='border-radius:12px 12px 0 0;border:1px solid var(--ink6);margin-bottom:0;'><span class='rp-head-title'>Radiograph</span></div>", unsafe_allow_html=True)
        heatmap = result.get("heatmap")
        if show_heatmap and heatmap:
            st.image(heatmap, caption="GradCAM — activation regions", use_container_width=True)
        elif img:
            st.image(img, caption="Uploaded radiograph", use_container_width=True)

        # Findings — built as single HTML block so divs nest correctly
        threshold_label = f"Flagged Pathologies &nbsp;<span style='font-weight:500;color:#c4cdd8;letter-spacing:0;text-transform:none;font-size:.65rem;'>threshold {threshold:.2f}</span>"
        findings_html = f"<div class='rp'><div class='rp-head'><span class='rp-head-title'>{threshold_label}</span></div><div class='rp-body'>"

        if not flagged:
            findings_html += f"<div style='text-align:center;padding:20px;font-size:.82rem;color:#8a97a8;'>No findings above threshold {threshold:.2f}. Lower the threshold in the sidebar.</div>"

        for i, f in enumerate(flagged):
            p = f["prob"]
            if p>=.75:   fc,bc,cc,ct = "fc-c","#dc2626","chip-c","Critical"
            elif p>=.45: fc,bc,cc,ct = "fc-m","#d97706","chip-m","Moderate"
            else:        fc,bc,cc,ct = "fc-l","#2563eb","chip-l","Low"
            findings_html += f"""
            <div class='fc {fc}' style='animation-delay:{i*60}ms'>
              <div style='display:flex;justify-content:space-between;align-items:center;'>
                <span class='fc-name'>{f['label']}</span>
                <span class='chip {cc}'>{ct}</span>
              </div>
              <div class='fc-conf'>confidence &nbsp;{p*100:.2f}%</div>
              <div class='bt'><div class='bf' style='width:{p*100:.1f}%;background:{bc};'></div></div>
            </div>"""

        if suppressed:
            findings_html += "<div style='margin-top:12px;font-size:.58rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#c4cdd8;margin-bottom:6px;'>Below Threshold</div>"
            for f in suppressed:
                findings_html += f"<div class='sup'><span class='sup-n'>{f['label']}</span><span class='sup-p'>{f['prob']*100:.1f}%</span></div>"

        ent = result.get("clinical_entities",{})
        if any(ent.values()):
            findings_html += "<div style='margin-top:14px;font-size:.58rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#c4cdd8;margin-bottom:8px;'>Clinical Entities</div>"
            tags = []
            if ent.get("age"):    tags.append(f"Age {ent['age']} y")
            if ent.get("gender"): tags.append(ent["gender"])
            for s in ent.get("symptoms",[]): tags.append(s.title())
            findings_html += "".join(f"<span class='etag'>{t}</span>" for t in tags)

        findings_html += "</div></div>"
        st.markdown(findings_html, unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class='rp'>
          <div class='rp-head'>
            <span class='rp-head-title'>AI-Generated Clinical Report</span>
            <span class='urg {uc}'>{ut}</span>
          </div>
          <div class='rp-body'>
            <div class='report'>{result['clinical_report']}</div>
          </div>
        </div>""", unsafe_allow_html=True)
        st.download_button("Download Report (.txt)", data=result["clinical_report"],
                           file_name="clinicalai_report.txt", mime="text/plain", use_container_width=True)
        st.markdown("""
        <div class='disc'>
          <strong>Clinical Disclaimer</strong> — This AI system is for research and demonstration
          purposes only. All findings must be verified by a licensed radiologist prior to any
          clinical decision-making. Not for diagnostic use.
        </div>""", unsafe_allow_html=True)

    # Chart
    st.markdown("<div class='div'></div>", unsafe_allow_html=True)
    st.markdown("<div class='st'>Pathology Probability Distribution</div>", unsafe_allow_html=True)

    import plotly.graph_objects as go
    ALL = list(AUC_DATA.keys())
    pd_ = {f["label"]: f["prob"] for f in findings}
    probs = [pd_.get(l, float(np.random.uniform(.02,.12))) for l in ALL]
    bc_   = ["#dc2626" if p>=.75 else ("#2563eb" if p>=threshold else "#e8ecf0") for p in probs]
    tc_   = ["#dc2626" if p>=.75 else ("#2563eb" if p>=threshold else "#c4cdd8") for p in probs]

    fig = go.Figure(go.Bar(
        x=probs, y=ALL, orientation="h",
        marker=dict(color=bc_, line=dict(width=0), opacity=.9, cornerradius=4),
        text=[f"{p*100:.1f}%" for p in probs], textposition="outside",
        textfont=dict(color=tc_, size=10.5, family="JetBrains Mono"),
        hovertemplate="<b>%{y}</b><br>Confidence: %{x:.1%}<extra></extra>",
        cliponaxis=False,
    ))
    fig.add_vline(x=threshold, line=dict(color="rgba(8,15,30,.18)", width=1.5, dash="dot"),
                  annotation_text=f"threshold {threshold:.2f}",
                  annotation_position="top",
                  annotation_font=dict(size=10, color="#8a97a8", family="JetBrains Mono"))
    fig.update_layout(
        height=460, margin=dict(l=0,r=72,t=12,b=12),
        xaxis=dict(range=[0,1.24], showgrid=True, gridcolor="rgba(8,15,30,.05)", gridwidth=1,
                   zeroline=True, zerolinecolor="rgba(8,15,30,.08)", zerolinewidth=1,
                   tickformat=".0%", tickfont=dict(color="#c4cdd8",size=10,family="JetBrains Mono"), title=None),
        yaxis=dict(tickfont=dict(color="#4b5768",size=11,family="Inter"), ticksuffix="  "),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False, bargap=.40,
        hoverlabel=dict(bgcolor="white",bordercolor="#e8ecf0",font=dict(color="#080f1e",size=12,family="Inter")),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    st.markdown("""
    <div class='chart-legend'>
      <span><span style='width:10px;height:10px;border-radius:3px;background:#dc2626;display:inline-block;'></span>Critical (&ge;75%)</span>
      <span><span style='width:10px;height:10px;border-radius:3px;background:#2563eb;display:inline-block;'></span>Above threshold</span>
      <span><span style='width:10px;height:10px;border-radius:3px;background:#e8ecf0;display:inline-block;'></span>Sub-threshold</span>
    </div>""", unsafe_allow_html=True)



def _live_predict(image, note, config):
    pipeline, error = load_pipeline(config)
    if error:
        st.error(f"Model loading failed: {error}"); st.stop()
    pred = pipeline.predict(image, note)
    return {"findings":pred.findings,"urgency_score":pred.urgency_score,
            "clinical_report":pred.clinical_report,"inference_time_ms":pred.inference_time_ms,
            "clinical_entities":pred.clinical_entities,"heatmap":pred.heatmap}


if __name__ == "__main__":
    main()
