#!/usr/bin/env python3
"""
Architecture diagram generator -- AI AutoML Intelligence Platform.
Produces static SVG diagrams in docs/assets/diagrams/, replacing the
renderer-dependent Mermaid blocks previously embedded in README.md.

Run from the repository root: python docs/assets/diagrams/source/generate_svgs.py

Vertical layout contract (all diagrams):
  0-46      main title band
  58        first zone top (12px clearance under title band)
  zone top + 0-30    reserved heading band (label baseline at zone top + 19)
  zone top + 30      first node row
  zone bottom - 12   content bottom padding
  canvas H - 16      last content bottom margin
"""

import pathlib

OUT = pathlib.Path("docs/assets/diagrams")
F   = "'Segoe UI','Helvetica Neue',Arial,sans-serif"
ARR = "#475569"
BG  = "#F8FAFC"
BD  = "#CBD5E1"
TBG = "#0F172A"

# Palette: (fill, stroke, text) -- teal-led palette matching the platform's
# existing brand accent (teal.300/teal.400 in the Next.js UI).
P = {
    "frontend":   ("#CCFBF1","#0D9488","#134E4A"),
    "route":      ("#DBEAFE","#2563EB","#1E3A8A"),
    "agent":      ("#DDD6FE","#7C3AED","#3B0764"),
    "orch":       ("#FEF9C3","#A16207","#713F12"),
    "rag":        ("#FED7AA","#EA580C","#7C2D12"),
    "data":       ("#D1FAE5","#059669","#064E3B"),
    "ai":         ("#FEE2E2","#B91C1C","#7F1D1D"),
    "input":      ("#DBEAFE","#1D4ED8","#1E40AF"),
    "output":     ("#CCFBF1","#0F766E","#134E4A"),
    "fusion":     ("#E2E8F0","#475569","#0F172A"),
}


def xe(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def hdr(w, h, title, desc):
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">\n'
            f'  <title>{xe(title)}</title>\n'
            f'  <desc>{xe(desc)}</desc>\n'
            f'  <defs>\n'
            f'    <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5"\n'
            f'            markerWidth="6" markerHeight="6" orient="auto">\n'
            f'      <polygon points="0,0 10,5 0,10" fill="{ARR}"/>\n'
            f'    </marker>\n'
            f'  </defs>\n'
            f'  <rect width="{w}" height="{h}" rx="8" fill="{BG}" stroke="{BD}" stroke-width="1.5"/>\n'
            f'  <rect x="0" y="0" width="{w}" height="46" rx="8" fill="{TBG}"/>\n'
            f'  <rect x="0" y="34" width="{w}" height="12" fill="{TBG}"/>\n'
            f'  <text x="{w // 2}" y="30" text-anchor="middle" font-family="{F}" '
            f'font-size="20" font-weight="700" fill="#F1F5F9">{xe(title)}</text>\n')


def nd(x, y, w, h, label, sub="", sty="fusion", rx=6, font=14):
    f_, s_, t_ = P[sty]
    o = f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{f_}" stroke="{s_}" stroke-width="1.5"/>\n'
    if sub:
        o += (f'  <text x="{x + w // 2}" y="{y + h // 2 - 5}" text-anchor="middle" '
              f'font-family="{F}" font-size="{font}" font-weight="600" fill="{t_}">{xe(label)}</text>\n'
              f'  <text x="{x + w // 2}" y="{y + h // 2 + 11}" text-anchor="middle" '
              f'font-family="{F}" font-size="11" fill="{t_}" opacity="0.85">{xe(sub)}</text>\n')
    else:
        o += (f'  <text x="{x + w // 2}" y="{y + h // 2 + 5}" text-anchor="middle" '
              f'font-family="{F}" font-size="{font}" font-weight="600" fill="{t_}">{xe(label)}</text>\n')
    return o


def zl(x, y, label, col="#64748B"):
    return (f'  <text x="{x}" y="{y}" font-family="{F}" font-size="10" font-weight="700" '
            f'fill="{col}" letter-spacing="1.4">{xe(label.upper())}</text>\n')


def ln(x1, y1, x2, y2, dash=False):
    d = ' stroke-dasharray="6,3"' if dash else ''
    return f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{ARR}" stroke-width="2"{d} marker-end="url(#ah)"/>\n'


def pa(d, dash=False):
    da = ' stroke-dasharray="6,3"' if dash else ''
    return f'  <path d="{d}" stroke="{ARR}" stroke-width="2" fill="none"{da} marker-end="url(#ah)"/>\n'


def cbez(x1, y1, cx1, cy1, cx2, cy2, x2, y2, dash=False):
    return pa(f"M{x1},{y1} C{cx1},{cy1} {cx2},{cy2} {x2},{y2}", dash)


def ftr():
    return '</svg>\n'


# ---------------------------------------------------------------------------
# Diagram 1 -- Platform Role in the ML Workflow
# ---------------------------------------------------------------------------
def d1():
    W, H = 1400, 430
    o = hdr(W, H, "Platform Role in the ML Workflow",
            "The AI AutoML Intelligence Platform sits between a raw CSV input, a local "
            "knowledge base, and Ollama, producing processed datasets, EDA reports, "
            "trained models, forecasts, AI insights, and full reports.")

    # Zone: Inputs
    o += f'  <rect x="10" y="58" width="1380" height="98" rx="4" fill="#F0FDFA"/>\n'
    o += zl(20, 77, "Inputs", "#0D9488")

    in_w, in_h, in_y = 380, 58, 88
    in_x = [100, 510, 920]
    o += nd(in_x[0], in_y, in_w, in_h, "Raw CSV Dataset", "User-uploaded tabular data", "input")
    o += nd(in_x[1], in_y, in_w, in_h, "Local Knowledge Base", "data_1/ -- FAISS-indexed documents", "rag")
    o += nd(in_x[2], in_y, in_w, in_h, "Ollama Local LLM", "mistral / gemma2 / llama3.3", "ai")

    # Hub
    hub_x, hub_y, hub_w, hub_h = 450, 180, 500, 70
    o += nd(hub_x, hub_y, hub_w, hub_h, "AI AutoML Intelligence Platform",
            "FastAPI backend + Next.js frontend", "fusion", font=16)

    # Zone: Outputs
    out_top = 274
    o += f'  <rect x="10" y="{out_top}" width="1380" height="106" rx="4" fill="#F0FDF4"/>\n'
    o += zl(20, out_top + 19, "Generated Outputs", "#15803D")

    out_w, out_h, out_y = 210, 64, out_top + 30
    out_x = [20 + i * 230 for i in range(6)]
    outputs = [
        ("Processed Dataset", "processed_data/"),
        ("EDA Report & Figures", "reports/figures/"),
        ("Models & SHAP Plots", "models/, reports/"),
        ("Forecast Chart & CSV", "reports/"),
        ("AI Insights via Ollama", "reports/ai_insights.txt"),
        ("Full HTML/PDF Report", "reports/full_report.*"),
    ]
    for x, (label, sub) in zip(out_x, outputs):
        o += nd(x, out_y, out_w, out_h, label, sub, "data", font=11)

    # Arrows: inputs -> hub
    hub_cx = hub_x + hub_w // 2
    for x in in_x:
        cx = x + in_w // 2
        o += cbez(cx, in_y + in_h, cx, in_y + in_h + 16, hub_cx, hub_y - 16, hub_cx, hub_y)

    # Arrows: hub -> outputs
    for x in out_x:
        cx = x + out_w // 2
        o += cbez(hub_cx, hub_y + hub_h, hub_cx, hub_y + hub_h + 18, cx, out_y - 16, cx, out_y)

    o += ftr()
    return o


# ---------------------------------------------------------------------------
# Diagram 2 -- System Architecture
# ---------------------------------------------------------------------------
def d2():
    W, H = 1400, 610
    o = hdr(W, H, "System Architecture",
            "Ten pipeline stages flow from Next.js frontend pages through FastAPI routes "
            "into dedicated agent modules, with a FAISS + SentenceTransformer + Ollama "
            "retrieval layer and generated artifacts written to local directories.")

    cw, gap = 110, 13
    step = cw + gap
    x0 = 20

    def col_x(i):
        return x0 + i * step

    def col_c(i):
        return col_x(i) + cw // 2

    stages = ["Preprocessing", "EDA", "Feature Eng.", "Model Training", "Evaluation",
              "Forecasting", "AI Insights", "RAG", "Orchestrator", "Reports", "Reset"]
    ui_cols = {0, 1, 3, 4, 5, 6, 7, 9}
    ui_routes = {0: "/preprocessing", 1: "/eda", 3: "/model-training", 4: "/evaluate",
                 5: "/forecasting", 6: "/insights", 7: "/rag", 9: "/reports"}
    agent_roles = ["agent", "agent", "agent", "agent", "agent", "agent",
                   "agent", "layer", "agent", "assembler", "utility"]

    # Zone 1: Frontend -- each node names the pipeline stage plus its route path.
    z1_top, z1_h = 58, 96
    o += f'  <rect x="10" y="{z1_top}" width="1380" height="{z1_h}" rx="4" fill="#F0FDFA"/>\n'
    o += zl(20, z1_top + 19, "Frontend -- Next.js", "#0D9488")
    ui_y, ui_h = z1_top + 30, 54
    for i in ui_cols:
        o += nd(col_x(i), ui_y, cw, ui_h, stages[i], ui_routes[i], "frontend", font=11)

    # Zone 2: Backend routes -- one FastAPI route module per pipeline stage
    # (full endpoint paths are listed in the AutoML Workflow table in README.md).
    z2_top, z2_h = z1_top + z1_h + 10, 98
    o += f'  <rect x="10" y="{z2_top}" width="1380" height="{z2_h}" rx="4" fill="#EFF6FF"/>\n'
    o += zl(20, z2_top + 19, "Backend -- API Routes", "#2563EB")
    rt_y, rt_h = z2_top + 30, 56
    for i in range(11):
        o += nd(col_x(i), rt_y, cw, rt_h, stages[i], "", "route", font=11)

    # Zone 3: Agents / orchestration -- one agent module per pipeline stage.
    z3_top, z3_h = z2_top + z2_h + 10, 98
    o += f'  <rect x="10" y="{z3_top}" width="1380" height="{z3_h}" rx="4" fill="#F5F3FF"/>\n'
    o += zl(20, z3_top + 19, "Backend -- Agents and Orchestration", "#7C3AED")
    ag_y, ag_h = z3_top + 30, 56
    for i in range(11):
        sty = "orch" if i == 8 else "agent"
        o += nd(col_x(i), ag_y, cw, ag_h, stages[i], agent_roles[i], sty, font=11)

    # Zone 4: RAG layer
    z4_top, z4_h = z3_top + z3_h + 10, 104
    o += f'  <rect x="10" y="{z4_top}" width="1380" height="{z4_h}" rx="4" fill="#FFF7ED"/>\n'
    o += zl(20, z4_top + 19, "RAG / Retrieval Layer", "#EA580C")
    rag_y, rag_h, rag_w, rag_gap = z4_top + 30, 60, 140, 20
    rag_x0 = col_x(6)
    rag_x = [rag_x0 + i * (rag_w + rag_gap) for i in range(4)]
    rag_nodes = [
        ("Knowledge Base", "data_1/ documents"),
        ("SentenceTransformer", "all-MiniLM-L6-v2"),
        ("FAISS Index", "k-NN vector search"),
        ("Ollama Local LLM", "Local LLM runtime"),
    ]
    for x, (label, sub) in zip(rag_x, rag_nodes):
        o += nd(x, rag_y, rag_w, rag_h, label, sub, "rag", font=11)

    # Zone 5: Artifacts
    z5_top, z5_h = z4_top + z4_h + 10, 100
    o += f'  <rect x="10" y="{z5_top}" width="1380" height="{z5_h}" rx="4" fill="#F0FDF4"/>\n'
    o += zl(20, z5_top + 19, "Generated Artifacts", "#15803D")
    art_y, art_h, art_w = z5_top + 30, 58, 300
    art_gap = (1360 - 4 * art_w) // 3
    art_x = [20 + i * (art_w + art_gap) for i in range(4)]
    art_nodes = [
        ("processed_data/", "Preprocessed CSVs (UUID-keyed)"),
        ("models/", "Trained PKL files"),
        ("reports/", "EDA figures, SHAP plots, forecast charts"),
        ("full_report.html / .pdf", "Assembled report"),
    ]
    for x, (label, sub) in zip(art_x, art_nodes):
        o += nd(x, art_y, art_w, art_h, label, sub, "data", font=13)

    # --- Arrows -------------------------------------------------------------
    # Frontend -> Routes (straight, column-aligned)
    for i in ui_cols:
        cx = col_c(i)
        o += ln(cx, ui_y + ui_h, cx, rt_y)
    # Routes -> Agents (straight, column-aligned)
    for i in range(11):
        cx = col_c(i)
        o += ln(cx, rt_y + rt_h, cx, ag_y)

    # RAG chain: Knowledge Base -> SentenceTransformer -> FAISS -> Ollama
    for i in range(3):
        y = rag_y + rag_h // 2
        o += ln(rag_x[i] + rag_w, y, rag_x[i + 1], y)
    # Agents -> RAG entry points (RAG layer agent -> Knowledge Base)
    o += cbez(col_c(7), ag_y + ag_h, col_c(7), ag_y + ag_h + 16,
              rag_x[0] + rag_w // 2, rag_y - 16, rag_x[0] + rag_w // 2, rag_y)
    # AI insights agent -> Ollama (direct dependency, dashed: stays above the
    # RAG row until the final approach so it never crosses SentenceTransformer/FAISS)
    ol_cx = rag_x[3] + rag_w // 2
    o += pa(f"M{col_c(6)},{ag_y + ag_h} C {col_c(6)},{rag_y - 15} {ol_cx},{rag_y - 15} {ol_cx},{rag_y}", dash=True)

    # Agents -> Artifacts (curves kept left/low until final approach to avoid
    # crossing the RAG layer nodes, which occupy x >= rag_x0)
    def agent_to_artifact(col_i, art_i, lift=140):
        sx, sy = col_c(col_i), ag_y + ag_h
        tx, ty = art_x[art_i] + art_w // 2, art_y
        return pa(f"M{sx},{sy} C {sx},{sy + lift} {tx},{ty - 30} {tx},{ty}")

    o += agent_to_artifact(0, 0)   # Preprocessing agent -> processed_data/
    o += agent_to_artifact(3, 1)   # Model training agent -> models/
    o += agent_to_artifact(1, 2)   # EDA agent -> reports/
    o += agent_to_artifact(3, 2)   # Model training agent -> reports/
    o += agent_to_artifact(5, 2)   # Forecasting agent -> reports/
    o += agent_to_artifact(9, 3)   # Report assembler -> full_report.*

    o += ftr()
    return o


def main():
    diagrams = [
        ("01_role_in_workflow.svg", d1),
        ("02_system_architecture.svg", d2),
    ]
    for fname, fn in diagrams:
        path = OUT / fname
        svg = fn()
        path.write_text(svg, encoding="utf-8")
        size = len(svg.encode("utf-8"))
        print(f"  wrote {path}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
