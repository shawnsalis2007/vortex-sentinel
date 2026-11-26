"""Vortex Sentinel — Shared Streamlit Theme, Navigation, Caching, and Plotly Styling.

Implements Vortex Sentinel – SOC Design System v1.0.4 tokens and micro-interactions.
"""

from __future__ import annotations

import html
from pathlib import Path
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "vortex_sentinel_logo.png"

from src.ml_engine import ModelBundle, load_or_train_model_bundle
from src.reports_and_db import ScanHistoryDB

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    :root {
        --bg-obsidian: #04080E;
        --card-slate: rgba(11, 20, 32, 0.90);
        --border-subtle: rgba(56, 189, 248, 0.14);
        --border-glow: rgba(56, 189, 248, 0.34);
        --primary-cyan: #38BDF8;
        --verified-teal: #2DD4BF;
        --medium-amber: #F59E0B;
        --critical-coral: #F43F5E;
        --ml-violet: #8B5CF6;
        --text-primary: #E2E8F0;
        --text-secondary: #CBD5E1;
        --text-muted: #94A3B8;
        --slate-500: #64748B;
        --text-on-primary: #070D14;
    }

    html, body, .stApp,
    h1, h2, h3, h4, h5, h6,
    p, li, label, input, textarea, select,
    table, th, td,
    .stButton > button p,
    .stDownloadButton > button p,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stDataFrame"],
    .soc-card, .vortex-card, .vortex-card-cyan,
    .vortex-bento, .vortex-bento-cyan, .vortex-hero,
    .vortex-bento-header, .vortex-kpi-label, .vortex-kpi-value,
    .vortex-badge, .vortex-badge-cyan, .vortex-pill,
    .status-pill, .vortex-risk-hero {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    h1 {
        letter-spacing: -0.5px;
        font-weight: 700;
    }

    h2, h3 {
        letter-spacing: -0.25px;
        font-weight: 600;
    }

    code, pre, .vortex-mono {
        font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
    }

    /* Strictly preserve Streamlit Material Icons / Symbols ligatures */
    span[data-testid="stIconMaterial"],
    [data-testid="stIconMaterial"],
    .material-icons,
    [class*="material-symbols"] {
        font-family: 'Material Symbols Rounded', 'Material Icons', 'Material Symbols Outlined' !important;
        font-feature-settings: 'liga' 1;
    }

    html, body {
        background: radial-gradient(circle at 50% 45%, #04080E 0%, #020408 100%) !important;
        background-color: #04080E !important;
        color: var(--text-primary);
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stAppViewContainer"] > .main {
        background: transparent !important;
        color: var(--text-primary);
        position: relative;
        z-index: 1;
    }

    section[data-testid="stSidebar"] {
        background-color: rgba(6, 11, 19, 0.94);
        z-index: 2;
    }

    iframe[height="0"] {
        position: fixed !important;
        width: 0 !important;
        height: 0 !important;
        border: 0 !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }

    .block-container {
        padding-top: 1.35rem;
        padding-bottom: 1.5rem;
        max-width: 1440px;
    }

    /* GPU-friendly non-interactive ambient SOC background layer */
    .vortex-ambient-bg {
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }

    .vortex-world-map {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        opacity: 0.78;
        pointer-events: none;
    }

    .vortex-ambient-scrim {
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at 50% 45%, rgba(4, 8, 14, 0.08) 0%, rgba(2, 4, 8, 0.58) 100%);
        pointer-events: none;
    }

    .vortex-ambient-grid {
        position: absolute;
        inset: -48px;
        background-image:
            linear-gradient(to right, rgba(56, 189, 248, 0.035) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(45, 212, 191, 0.03) 1px, transparent 1px);
        background-size: 48px 48px;
        will-change: transform, opacity;
        animation: vortexGridDrift 28s ease-in-out infinite alternate;
    }

    .vortex-ambient-orb-cyan {
        position: absolute;
        top: -120px;
        left: -80px;
        width: 520px;
        height: 520px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.11) 0%, rgba(45, 212, 191, 0.04) 45%, transparent 72%);
        will-change: transform, opacity;
        animation: vortexOrbCyan 20s ease-in-out infinite alternate;
    }

    .vortex-ambient-orb-violet {
        position: absolute;
        bottom: -140px;
        right: -90px;
        width: 560px;
        height: 560px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.11) 0%, rgba(56, 189, 248, 0.04) 48%, transparent 74%);
        will-change: transform, opacity;
        animation: vortexOrbViolet 24s ease-in-out infinite alternate;
    }

    @keyframes vortexGridDrift {
        0% { transform: translate3d(0px, 0px, 0); opacity: 0.55; }
        100% { transform: translate3d(24px, 24px, 0); opacity: 0.85; }
    }

    @keyframes vortexOrbCyan {
        0% { transform: translate3d(0px, 0px, 0) scale(1); opacity: 0.65; }
        100% { transform: translate3d(80px, 50px, 0) scale(1.08); opacity: 0.95; }
    }

    @keyframes vortexOrbViolet {
        0% { transform: translate3d(0px, 0px, 0) scale(1); opacity: 0.60; }
        100% { transform: translate3d(-70px, -45px, 0) scale(1.08); opacity: 0.90; }
    }

    @keyframes pulse-dot {
        0% { opacity: 0.4; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.2); }
        100% { opacity: 0.4; transform: scale(1); }
    }

    .status-dot, .vortex-status-dot {
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: var(--verified-teal);
        margin-right: 6px;
        vertical-align: middle;
        animation: pulse-dot 2000ms infinite;
    }

    header[data-testid="stHeader"] {
        background-color: rgba(7, 13, 20, 0.88);
        backdrop-filter: blur(8px);
        border-bottom: 1px solid var(--border-subtle);
    }

    section[data-testid="stSidebar"] {
        background-color: #09111B;
        border-right: 1px solid var(--border-subtle);
    }

    /* Hide default Streamlit page list so we render clean custom labels (no raw 'app') */
    [data-testid="stSidebarNav"] {
        display: none;
    }

    /* Sidebar nav link polish */
    section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {
        border-radius: 0.5rem;
        padding: 0.42rem 0.68rem;
        margin-bottom: 2px;
        border-left: 2px solid transparent;
        transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);
    }

    section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {
        background-color: var(--card-slate);
        border-left-color: var(--primary-cyan);
    }

    section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] {
        background: rgba(56, 189, 248, 0.12);
        border-left-color: var(--primary-cyan);
        font-weight: 600;
    }

    /* SOC Design System v1.0.4 Cards & Containers */
    .vortex-hero {
        background-color: var(--card-slate);
        border: 1px solid var(--border-subtle);
        border-radius: 1rem;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);
    }

    .vortex-hero:hover {
        border-color: var(--border-glow);
        box-shadow: 0 0 15px -5px var(--border-glow);
    }

    .soc-card, .vortex-card, .vortex-bento, .vortex-card-cyan, .vortex-bento-cyan {
        background-color: var(--card-slate);
        border: 1px solid var(--border-subtle);
        border-radius: 1rem;
        padding: 14px 16px;
        margin-bottom: 10px;
        transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);
    }

    .soc-card:hover, .vortex-bento:hover, .vortex-bento-cyan:hover,
    .vortex-card:hover, .vortex-card-cyan:hover {
        border-color: var(--border-glow);
        box-shadow: 0 0 15px -5px var(--border-glow);
        transform: translateY(-2px);
    }

    .vortex-bento-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
        padding-bottom: 6px;
        margin-bottom: 8px;
        border-bottom: 1px solid var(--border-subtle);
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--text-muted);
    }

    .vortex-telemetry-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
        margin-bottom: 10px;
    }

    /* Tertiary / ML Violet Badge */
    .vortex-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 11px;
        border-radius: 9999px;
        font-size: 0.73rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        background: rgba(139, 92, 246, 0.2);
        color: var(--ml-violet);
        border: 1px solid rgba(139, 92, 246, 0.4);
    }

    /* Verified / Cyan-Teal Badge */
    .vortex-badge-cyan {
        display: inline-flex;
        align-items: center;
        padding: 4px 11px;
        border-radius: 9999px;
        font-size: 0.73rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        background: rgba(45, 212, 191, 0.1);
        color: var(--verified-teal);
        border: 1px solid rgba(45, 212, 191, 0.2);
    }

    .vortex-kpi-label {
        font-size: 0.74rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        line-height: 1.35;
        word-break: normal;
        overflow-wrap: normal;
        white-space: normal;
    }

    .vortex-kpi-value {
        font-size: clamp(1.35rem, 2.2vw, 1.85rem);
        font-weight: 700;
        font-variant-numeric: tabular-nums;
        color: var(--primary-cyan);
        margin-top: 6px;
        line-height: 1.1;
    }

    /* Dominant Risk Score Hero Card */
    .vortex-risk-hero {
        background-color: var(--card-slate);
        border: 1px solid var(--border-subtle);
        border-radius: 1rem;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);
    }

    .vortex-risk-hero:hover {
        border-color: var(--border-glow);
        box-shadow: 0 0 15px -5px var(--border-glow);
    }

    .vortex-risk-bar-track {
        width: 100%;
        height: 8px;
        background: var(--bg-obsidian);
        border-radius: 9999px;
        overflow: hidden;
        margin-top: 10px;
        border: 1px solid var(--border-subtle);
    }

    .vortex-risk-bar-fill {
        height: 100%;
        border-radius: 9999px;
    }

    /* Status Pills */
    .status-pill, .vortex-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        border: 1px solid;
        white-space: nowrap;
    }
    .status-pill.verified, .vortex-pill-low {
        background: rgba(45, 212, 191, 0.1);
        color: var(--verified-teal);
        border-color: rgba(45, 212, 191, 0.2);
    }
    .vortex-pill-medium {
        background: rgba(56, 189, 248, 0.1);
        color: var(--primary-cyan);
        border-color: rgba(56, 189, 248, 0.25);
    }
    .status-pill.medium, .vortex-pill-high {
        background: rgba(245, 158, 11, 0.1);
        color: var(--medium-amber);
        border-color: rgba(245, 158, 11, 0.2);
    }
    .status-pill.critical, .vortex-pill-critical {
        background: rgba(244, 63, 94, 0.1);
        color: var(--critical-coral);
        border-color: rgba(244, 63, 94, 0.2);
    }
    .vortex-pill-unsupported {
        background: rgba(100, 116, 139, 0.14);
        color: var(--text-secondary);
        border-color: rgba(100, 116, 139, 0.28);
    }

    .vortex-mono {
        background: var(--bg-obsidian);
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        padding: 2px 7px;
        font-size: 0.76rem;
        line-height: 1.6;
        color: var(--primary-cyan);
    }

    div[data-testid="stMetric"] {
        background-color: var(--card-slate);
        border: 1px solid var(--border-subtle);
        border-radius: 1rem;
        padding: 12px 16px;
        transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);
    }

    div[data-testid="stMetric"]:hover {
        border-color: var(--border-glow);
        box-shadow: 0 0 15px -5px var(--border-glow);
        transform: translateY(-2px);
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: var(--card-slate);
        border: 1px dashed rgba(56, 189, 248, 0.28);
        border-radius: 1rem;
        padding: 1.1rem;
        transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--primary-cyan);
        box-shadow: 0 0 15px -5px var(--border-glow);
    }

    [data-testid="stExpander"] {
        background-color: var(--card-slate);
        border: 1px solid var(--border-subtle);
        border-radius: 1rem;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-subtle);
        border-radius: 1rem;
        overflow-x: auto;
    }

    /* Buttons (SOC Design System v1.0.4) */
    .stButton > button, .stDownloadButton > button {
        border-radius: 0.65rem;
        padding: 0.45rem 0.9rem;
        min-height: 2.5rem;
        border: 1px solid var(--text-primary);
        background: transparent;
        color: var(--text-primary);
        transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--primary-cyan);
        color: var(--primary-cyan);
        box-shadow: 0 0 15px -5px var(--border-glow);
    }

    .stButton > button p, .stDownloadButton > button p {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-size: 0.90rem;
    }

    .focus-ring:focus,
    .stButton > button:focus-visible,
    .stDownloadButton > button:focus-visible,
    [data-testid="stSidebar"] a:focus-visible {
        outline: 2px solid var(--primary-cyan);
        outline-offset: 2px;
    }

    .stButton > button[kind="primary"] {
        background-color: var(--primary-cyan);
        color: var(--text-on-primary);
        font-weight: 700;
        border: 1px solid var(--primary-cyan);
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #7DD3FC;
        border-color: #7DD3FC;
        color: var(--text-on-primary);
        box-shadow: 0 0 15px -3px var(--border-glow);
    }

    @media (max-width: 900px) {
        .block-container {
            padding-top: 0.9rem;
            padding-left: 0.85rem;
            padding-right: 0.85rem;
        }
        .vortex-hero, .vortex-bento, .vortex-bento-cyan, .vortex-risk-hero {
            padding: 12px 14px;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            transition: none !important;
            animation: none !important;
        }
    }
</style>
"""

AMBIENT_BG_HTML = """
<div class="vortex-ambient-bg" aria-hidden="true">
    <div class="vortex-ambient-orb-cyan"></div>
    <div class="vortex-ambient-orb-violet"></div>
    <div class="vortex-ambient-scrim"></div>
</div>
"""

TELEMETRY_CANVAS_SCRIPT = """
<script>
(function() {
  try {
    var pDoc = window.parent.document;
    var pWin = window.parent;
    if (!pDoc || !pDoc.body || !pDoc.head) return;

    var state = pWin.__VORTEX_BG_STATE__;
    if (
      state &&
      state.version === '1.0.8' &&
      state.canvas &&
      pDoc.body.contains(state.canvas) &&
      (state.reduceMotion || state.rafId)
    ) {
      state.canvas.style.position = 'fixed';
      state.canvas.style.inset = '0';
      state.canvas.style.zIndex = '0';
      state.canvas.style.pointerEvents = 'none';
      return;
    }

    if (state && state.rafId) {
      try { pWin.cancelAnimationFrame(state.rafId); } catch (_) {}
    }
    if (state && state.resizeHandler) {
      try { pWin.removeEventListener('resize', state.resizeHandler); } catch (_) {}
    }

    ['vortex-soc-canvas', 'vortex-bg-canvas', 'vortex-soc-runner'].forEach(function(id) {
      var el = pDoc.getElementById(id);
      if (el && el.parentNode) el.parentNode.removeChild(el);
    });

    var runner = pDoc.createElement('script');
    runner.id = 'vortex-soc-runner';
    runner.textContent = '(' + function() {
      var pDoc = document;
      var pWin = window;
      var canvas = pDoc.createElement('canvas');
      canvas.id = 'vortex-soc-canvas';
      canvas.setAttribute('data-vortex-version', '1.0.8');
      canvas.style.position = 'fixed';
      canvas.style.inset = '0';
      canvas.style.width = '100vw';
      canvas.style.height = '100vh';
      canvas.style.pointerEvents = 'none';
      canvas.style.zIndex = '0';
      canvas.style.opacity = '0.82';
      pDoc.body.appendChild(canvas);

      var ctx = canvas.getContext('2d');
      var mapCanvas = pDoc.createElement('canvas');
      var mapCtx = mapCanvas.getContext('2d');
      var reduceMotion = !!(pWin.matchMedia && pWin.matchMedia('(prefers-reduced-motion: reduce)').matches);
      var w = 0, h = 0;

      var bgState = {
        version: '1.0.8',
        canvas: canvas,
        rafId: null,
        reduceMotion: reduceMotion,
        resizeHandler: null
      };
      pWin.__VORTEX_BG_STATE__ = bgState;

    // High-vertex-count equirectangular (lon, lat) world continent & archipelago polygons
    var CONTINENTS = [
      // North America (Alaska, Yukon, Hudson Bay, Labrador, US East/West, Florida, Gulf, Mexico, Central America)
      [[-168,65],[-166,68],[-162,70],[-156,71],[-148,70],[-140,70],[-134,69],[-128,71],[-120,69],[-110,69],[-96,70],[-88,68],[-82,66],[-78,62],[-80,58],[-85,55],[-92,57],[-94,55],[-88,52],[-80,51],[-79,54],[-76,62],[-68,60],[-62,58],[-55,52],[-58,49],[-64,47],[-60,46],[-65,44],[-67,45],[-70,43],[-70,41],[-74,40],[-75,38],[-76,35],[-80,32],[-81,29],[-80,25],[-82,25],[-82,28],[-85,30],[-89,30],[-93,29],[-97,27],[-97,23],[-96,19],[-92,18],[-90,21],[-87,21],[-88,18],[-88,16],[-85,15],[-83,12],[-83,10],[-79,9],[-78,8],[-81,8],[-85,10],[-87,13],[-92,14],[-96,16],[-102,18],[-106,22],[-108,26],[-112,30],[-114,31],[-111,24],[-114,27],[-117,32],[-120,34],[-122,37],[-124,41],[-124,46],[-123,48],[-128,51],[-132,55],[-138,59],[-146,60],[-152,60],[-156,57],[-162,55],[-165,58],[-162,61],[-166,63]],
      // Cuba & Hispaniola
      [[-84,22],[-79,23],[-74,20],[-77,20],[-82,22]],
      [[-74,19],[-69,19],[-69,18],[-73,18]],
      // Greenland
      [[-54,60],[-45,60],[-40,65],[-32,68],[-22,71],[-19,76],[-22,81],[-32,83],[-46,82],[-58,80],[-68,76],[-60,72],[-54,67]],
      // Iceland
      [[-24,65],[-20,66],[-15,66],[-14,64],[-18,63],[-23,64]],
      // Great Britain & Ireland
      [[-5,50],[1,51],[1,53],[-1,55],[-2,58],[-5,58],[-6,56],[-4,54],[-5,52]],
      [[-10,52],[-8,51],[-6,52],[-6,54],[-8,55],[-10,54]],
      // South America (Colombia, Venezuela, Guiana, Amazon, Brazil Bulge, Rio, Patagonia, Andes, Peru, Ecuador)
      [[-78,9],[-75,11],[-72,12],[-68,10],[-62,10],[-56,6],[-50,1],[-45,-2],[-39,-4],[-35,-6],[-36,-10],[-39,-15],[-40,-21],[-44,-23],[-48,-27],[-51,-31],[-54,-34],[-58,-38],[-63,-41],[-65,-45],[-67,-49],[-69,-54],[-73,-53],[-75,-48],[-74,-42],[-72,-36],[-71,-30],[-70,-22],[-71,-16],[-76,-14],[-79,-8],[-81,-5],[-80,0],[-78,5]],
      // Europe (Iberia, France, Italy, Greece, Balkans, Black Sea, Baltic, Scandinavia)
      [[-9,37],[-9,41],[-8,43],[-2,43],[-2,47],[-5,48],[-1,49],[2,51],[5,53],[8,54],[8,57],[10,57],[12,56],[10,54],[14,54],[19,54],[21,57],[24,57],[25,59],[29,60],[27,62],[22,65],[18,60],[16,56],[13,56],[11,59],[8,58],[5,60],[6,63],[12,67],[18,70],[26,71],[31,70],[38,69],[43,67],[40,64],[34,66],[32,62],[29,60],[44,58],[48,55],[44,48],[39,47],[35,45],[31,46],[29,43],[29,41],[26,40],[24,38],[22,37],[20,39],[18,41],[14,45],[12,44],[14,41],[16,40],[16,38],[14,40],[11,43],[9,44],[5,43],[3,42],[0,39],[-3,37],[-6,36]],
      // Sicily & Sardinia
      [[12,38],[15,38],[15,37],[13,37]],
      // Africa (Morocco, Mediterranean, Red Sea, Horn, East Africa, Cape, Namibia, Gulf of Guinea, West Bulge)
      [[-17,15],[-16,20],[-13,27],[-9,32],[-6,36],[0,36],[8,37],[11,37],[11,34],[16,32],[20,31],[25,31],[32,31],[34,27],[37,22],[40,16],[43,12],[48,11],[51,12],[49,6],[45,2],[41,-2],[40,-8],[40,-14],[35,-20],[35,-25],[31,-30],[27,-33],[22,-34],[18,-34],[16,-28],[14,-21],[12,-14],[12,-6],[9,-1],[9,4],[4,6],[-2,5],[-8,5],[-13,8],[-16,12]],
      // Madagascar
      [[44,-25],[47,-25],[48,-20],[50,-15],[49,-12],[46,-16],[44,-21]],
      // Arabian Peninsula & Levant
      [[35,36],[36,31],[35,28],[39,22],[43,13],[46,13],[53,16],[59,22],[56,26],[52,24],[50,28],[48,30],[44,34],[39,37]],
      // Main Asia (Siberia, Kamchatka, China, Korea, Indochina, Malay, India, Iran)
      [[39,37],[45,40],[50,42],[53,46],[58,52],[62,58],[66,66],[72,72],[82,74],[95,77],[108,77],[122,74],[138,72],[156,70],[172,68],[179,66],[172,62],[162,60],[162,54],[157,51],[154,58],[144,58],[140,54],[141,49],[136,44],[132,42],[129,38],[129,35],[126,35],[126,39],[122,39],[122,37],[119,35],[122,31],[120,27],[117,24],[113,22],[109,21],[107,20],[109,15],[109,11],[105,9],[102,12],[100,13],[101,8],[104,2],[101,3],[98,9],[98,15],[95,16],[93,20],[90,22],[86,20],[82,16],[80,10],[77,8],[75,13],[72,19],[69,22],[67,25],[62,25],[57,26],[52,29],[48,30],[44,34]],
      // Sri Lanka & Taiwan
      [[80,9],[82,8],[81,6],[80,7]],
      [[120,25],[122,25],[121,22],[120,23]],
      // Japan (Kyushu, Honshu, Hokkaido)
      [[130,31],[132,33],[136,34],[140,36],[141,40],[140,41],[138,37],[134,35],[131,34]],
      [[140,42],[145,43],[144,45],[141,45],[140,43]],
      // Sumatra, Java, Borneo, Sulawesi, Philippines, New Guinea
      [[95,5],[99,3],[104,-2],[106,-6],[103,-6],[99,-1],[95,4]],
      [[106,-6],[114,-7],[114,-8],[106,-7]],
      [[109,1],[114,3],[118,6],[118,1],[116,-3],[111,-3],[109,-1]],
      [[119,1],[125,1],[123,-4],[120,-5],[119,-2]],
      [[120,18],[122,18],[123,13],[126,9],[125,6],[122,7],[121,12],[120,15]],
      [[131,-2],[136,-2],[141,-3],[146,-7],[150,-10],[146,-9],[141,-9],[136,-5],[132,-4]],
      // Australia (WA, Arnhem, Cape York, Queensland, NSW, Victoria, SA, Bight)
      [[114,-22],[119,-20],[123,-17],[129,-14],[131,-12],[136,-12],[136,-15],[140,-17],[142,-11],[145,-15],[149,-21],[153,-26],[153,-31],[150,-37],[145,-38],[140,-37],[137,-35],[134,-33],[128,-32],[122,-34],[116,-34],[114,-29],[113,-25]],
      // Tasmania
      [[145,-41],[148,-41],[148,-43],[145,-43]],
      // New Zealand (North & South Islands)
      [[173,-35],[178,-38],[176,-41],[174,-41],[174,-37]],
      [[172,-41],[174,-42],[170,-46],[167,-46],[170,-43]]
    ];

    function projectLonLat(lon, lat) {
      var x = ((lon + 180) / 360) * w;
      var y = ((82 - lat) / 140) * h;
      return [x, y];
    }

    function renderWorldMap() {
      mapCanvas.width = w;
      mapCanvas.height = h;
      mapCtx.clearRect(0, 0, w, h);

      var patCanvas = pDoc.createElement('canvas');
      patCanvas.width = 6;
      patCanvas.height = 6;
      var patCtx = patCanvas.getContext('2d');
      patCtx.fillStyle = 'rgba(56, 189, 248, 0.18)';
      patCtx.fillRect(1, 1, 1.3, 1.3);
      var dotPattern = mapCtx.createPattern(patCanvas, 'repeat');

      mapCtx.save();
      mapCtx.beginPath();
      for (var c = 0; c < CONTINENTS.length; c++) {
        var poly = CONTINENTS[c];
        for (var p = 0; p < poly.length; p++) {
          var pt = projectLonLat(poly[p][0], poly[p][1]);
          if (p === 0) mapCtx.moveTo(pt[0], pt[1]);
          else mapCtx.lineTo(pt[0], pt[1]);
        }
        mapCtx.closePath();
      }
      mapCtx.fillStyle = 'rgba(56, 189, 248, 0.09)';
      mapCtx.fill();
      if (dotPattern) {
        mapCtx.fillStyle = dotPattern;
        mapCtx.fill();
      }
      mapCtx.strokeStyle = 'rgba(56, 189, 248, 0.25)';
      mapCtx.lineWidth = 1.05;
      mapCtx.stroke();
      mapCtx.restore();
    }

    function resize() {
      w = canvas.width = pWin.innerWidth;
      h = canvas.height = pWin.innerHeight;
      renderWorldMap();
    }
    resize();
    bgState.resizeHandler = resize;
    pWin.addEventListener('resize', resize);

    var palette = [
      { color: '#38BDF8', xMin: 0.04, xMax: 0.42 },
      { color: '#38BDF8', xMin: 0.14, xMax: 0.52 },
      { color: '#F59E0B', xMin: 0.36, xMax: 0.74 },
      { color: '#F43F5E', xMin: 0.62, xMax: 0.96 }
    ];

    var nodes = [];
    for (var i = 0; i < 45; i++) {
      var p = palette[i % palette.length];
      nodes.push({
        xr: p.xMin + Math.random() * (p.xMax - p.xMin),
        yr: 0.08 + Math.random() * 0.84,
        vx: (Math.random() - 0.5) * 0.00135,
        vy: (Math.random() - 0.5) * 0.00135,
        r: 2.2 + Math.random() * 2.1,
        phase: Math.random() * Math.PI * 2,
        color: p.color,
        xMin: p.xMin,
        xMax: p.xMax
      });
    }

    var packets = [];
    for (var k = 0; k < 16; k++) {
      packets.push({ t: Math.random(), speed: 0.022 + Math.random() * 0.020, seed: k });
    }

    function draw() {
      if (!reduceMotion) bgState.rafId = pWin.requestAnimationFrame(draw);
      if (pDoc.hidden) return;

      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(mapCanvas, 0, 0);

      for (var i = 0; i < nodes.length; i++) {
        var n = nodes[i];
        if (!reduceMotion) {
          n.xr += n.vx;
          n.yr += n.vy;
          if (n.xr < n.xMin || n.xr > n.xMax) n.vx *= -1;
          if (n.yr < 0.06 || n.yr > 0.94) n.vy *= -1;
          n.phase += 0.055;
        }
        n.px = n.xr * w;
        n.py = n.yr * h;
      }

      var activeEdges = [];
      for (var i = 0; i < nodes.length; i++) {
        for (var j = i + 1; j < nodes.length; j++) {
          var a = nodes[i], b = nodes[j];
          var dx = a.px - b.px, dy = a.py - b.py;
          var dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 195) {
            var alpha = (1 - dist / 195) * 0.48;
            ctx.beginPath();
            ctx.moveTo(a.px, a.py);
            ctx.lineTo(b.px, b.py);
            ctx.strokeStyle = a.color;
            ctx.globalAlpha = alpha;
            ctx.lineWidth = 1.05;
            ctx.stroke();
            if (activeEdges.length < 48) {
              activeEdges.push([a, b]);
            }
          }
        }
      }

      if (!reduceMotion && activeEdges.length > 0) {
        for (var k = 0; k < packets.length; k++) {
          var pk = packets[k];
          pk.t += pk.speed;
          if (pk.t > 1) pk.t -= 1;
          var edge = activeEdges[(pk.seed * 7) % activeEdges.length];
          var px = edge[0].px + (edge[1].px - edge[0].px) * pk.t;
          var py = edge[0].py + (edge[1].py - edge[0].py) * pk.t;

          ctx.globalAlpha = 0.40;
          ctx.beginPath();
          ctx.arc(px, py, 5.2, 0, Math.PI * 2);
          ctx.fillStyle = edge[0].color;
          ctx.fill();

          ctx.globalAlpha = 0.95;
          ctx.beginPath();
          ctx.arc(px, py, 1.9, 0, Math.PI * 2);
          ctx.fillStyle = '#FFFFFF';
          ctx.fill();
        }
      }

      for (var i = 0; i < nodes.length; i++) {
        var n = nodes[i];
        var pulse = 0.65 + 0.35 * Math.sin(n.phase);
        var grad = ctx.createRadialGradient(n.px, n.py, 0, n.px, n.py, n.r * 4.8);
        grad.addColorStop(0, n.color);
        grad.addColorStop(1, 'rgba(4, 8, 14, 0)');
        ctx.globalAlpha = 0.42 * pulse;
        ctx.beginPath();
        ctx.arc(n.px, n.py, n.r * 4.8, 0, Math.PI * 2);
        ctx.fillStyle = grad;
        ctx.fill();

        ctx.globalAlpha = 0.92 * pulse;
        ctx.beginPath();
        ctx.arc(n.px, n.py, n.r, 0, Math.PI * 2);
        ctx.fillStyle = n.color;
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    if (reduceMotion) {
      draw();
    } else {
      bgState.rafId = pWin.requestAnimationFrame(draw);
    }
    }.toString() + ')();';
    pDoc.head.appendChild(runner);
  } catch (e) {}
})();
</script>
"""


def inject_vortex_theme() -> None:
    """Apply SOC Design System v1.0.4 styling and clean sidebar navigation."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(AMBIENT_BG_HTML, unsafe_allow_html=True)
    components.html(TELEMETRY_CANVAS_SCRIPT, height=0, width=0)
    with st.sidebar:
        if LOGO_PATH.exists():
            s_logo, s_title = st.columns([0.32, 0.68], vertical_alignment="center")
            with s_logo:
                st.image(str(LOGO_PATH), width=56)
            with s_title:
                st.markdown("### Vortex Sentinel")
        else:
            st.markdown("### 🛡️ Vortex Sentinel")
        st.divider()
        st.page_link("pages/0_Landing.py", label="🏠 Landing")
        st.page_link("app.py", label="📊 SOC Dashboard")
        st.page_link("pages/1_Static_Triage_Workbench.py", label="🧪 Unified Triage Workbench")
        st.page_link("pages/1_Executable_Scanner.py", label="🛡️ Executable Scanner")
        st.page_link("pages/2_PDF_Inspector.py", label="📄 PDF Inspector")
        st.page_link("pages/3_Batch_Analysis.py", label="📦 Batch Analysis")
        st.page_link("pages/4_Model_Comparison.py", label="⚖️ Model Comparison")
        st.page_link("pages/5_Explainable_AI.py", label="🔍 Explainable AI (SHAP)")
        st.page_link("pages/6_Investigation_Reports.py", label="📑 Investigation Reports")
        st.page_link("pages/7_Scan_History.py", label="🗂️ Scan History")


def render_page_header(
    title: str,
    subtitle: str,
    badge: str = "ZERO-EXECUTION STATIC ANALYSIS",
    status_tag: str = "TEAM VORTEX • AV26-4C8B10F965",
) -> None:
    """Render a compact, consistent SOC workstation header for sub-pages."""
    safe_title = html.escape(title)
    safe_sub = html.escape(subtitle)
    safe_badge = html.escape(badge)
    safe_tag = html.escape(status_tag)
    if LOGO_PATH.exists():
        hdr_logo, hdr_card = st.columns([0.08, 0.92], vertical_alignment="center")
        with hdr_logo:
            st.image(str(LOGO_PATH), width=76)
    else:
        hdr_card = st.container()

    with hdr_card:
        st.markdown(
            f"""
            <div class="vortex-hero" style="padding: 14px 18px; margin-bottom: 12px;">
                <div class="vortex-telemetry-strip" style="margin-bottom: 6px;">
                    <span class="vortex-badge-cyan"><span class="vortex-status-dot"></span>{safe_badge}</span>
                    <span class="vortex-badge">{safe_tag}</span>
                </div>
                <h2 style="margin: 0 0 4px 0; font-size: 1.5rem; font-weight: 700; color: #E2E8F0;">{safe_title}</h2>
                <p style="margin: 0; font-size: 0.90rem; color: #94A3B8; line-height: 1.5;">{safe_sub}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_risk_score_hero(
    risk_score: float | None,
    priority: str,
    label: str = "Estimated Static Risk Score",
    subtitle: str = "Probabilistic estimate (0–100) — not proof of malice or safety.",
) -> None:
    """Render the dominant risk score card with a proportional color-coded bar."""
    color_map = {
        "low": "#2DD4BF",
        "medium": "#38BDF8",
        "high": "#F59E0B",
        "critical": "#F43F5E",
    }
    accent = color_map.get(priority.strip().lower(), "#94A3B8")
    score_txt = f"{risk_score:.1f}" if risk_score is not None else "N/A"
    pct = max(0.0, min(100.0, float(risk_score))) if risk_score is not None else 0.0
    st.markdown(
        f"""
        <div class="vortex-risk-hero" style="border-color: {accent}44;">
            <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px;">
                <div>
                    <div class="vortex-kpi-label">{html.escape(label)}</div>
                    <div style="display: flex; align-items: baseline; gap: 12px; margin-top: 4px;">
                        <span style="font-size: 2.35rem; font-weight: 700; font-variant-numeric: tabular-nums; color: {accent}; line-height: 1;">
                            {score_txt}
                        </span>
                        <span style="font-size: 0.88rem; color: #94A3B8;">/ 100</span>
                        {render_priority_badge(priority)}
                    </div>
                </div>
                <div style="font-size: 0.82rem; color: #94A3B8; max-width: 360px;">
                    {html.escape(subtitle)}
                </div>
            </div>
            <div class="vortex-risk-bar-track">
                <div class="vortex-risk-bar-fill" style="width: {pct:.1f}%; background: {accent};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_priority_badge(priority: str) -> str:
    """Return HTML for a color-coded SOC priority pill badge with dot marker."""
    key = priority.strip().lower()
    css_class = {
        "low": "vortex-pill-low",
        "medium": "vortex-pill-medium",
        "high": "vortex-pill-high",
        "critical": "vortex-pill-critical",
    }.get(key, "vortex-pill-unsupported")
    dot_color = {
        "low": "#2DD4BF",
        "medium": "#38BDF8",
        "high": "#F59E0B",
        "critical": "#F43F5E",
    }.get(key, "#94A3B8")
    safe_label = html.escape(priority.upper())
    return (
        f'<span class="vortex-pill {css_class}">'
        f'<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:{dot_color};"></span>'
        f'{safe_label}</span>'
    )


@st.cache_resource
def get_model_bundle() -> ModelBundle:
    """Load persisted models under models/ or train synthetic benchmark models once."""
    return load_or_train_model_bundle()


@st.cache_resource
def get_db() -> ScanHistoryDB:
    """Return shared SQLite scan history database handle."""
    return ScanHistoryDB()


def apply_vortex_plotly_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply SOC Design System v1.0.4 styling to vertical Plotly charts."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#E2E8F0")) if title else "",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#E2E8F0",
            family="Plus Jakarta Sans, -apple-system, BlinkMacSystemFont, sans-serif",
        ),
        margin=dict(l=18, r=18, t=44, b=44),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(
            bgcolor="#070D14",
            bordercolor="#38BDF8",
            font=dict(color="#E2E8F0", family="Plus Jakarta Sans, sans-serif", size=13),
        ),
        transition=dict(duration=0),
    )
    fig.update_traces(marker_line_width=0)
    fig.update_xaxes(
        gridcolor="rgba(255, 255, 255, 0.05)",
        zerolinecolor="#1E293B",
        tickangle=-18,
    )
    fig.update_yaxes(
        gridcolor="rgba(56, 189, 248, 0.10)",
        zerolinecolor="#1E293B",
    )
    return fig
