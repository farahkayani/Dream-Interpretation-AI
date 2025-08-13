# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 04:08:52 2025

@author: farah
"""

# app.py — Streamlit Dream Interpreter (starter template)
# ------------------------------------------------------
# Features
# - Clean, single-file Streamlit app (multi-section via sidebar)
# - Load/merge/export a JSONL dream dictionary (id + text)
# - Interpret page with simple rule-based engine using your dictionary
# - Dictionary manager (search, add, edit, delete)
# - Analytics (quick word-frequency overview)
# - Persistent session state + local file saving
# - Gentle styling + light/dark theme support (via Streamlit theme)
# ------------------------------------------------------

from __future__ import annotations
import streamlit as st
import pandas as pd
from pathlib import Path
import json
import re
from collections import Counter
from datetime import datetime

APP_TITLE = "Dream Interpreter ✨"
APP_ICON = "💭"
DATA_DIR = Path("data")
DICT_PATH = DATA_DIR / "dream_dict.jsonl"
HISTORY_PATH = DATA_DIR / "history.jsonl"

# ---------------------------
# Utilities: JSONL I/O
# ---------------------------

def ensure_data_dir() -> None:
    DATA_DIR.mkdir(exist_ok=True)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # skip bad lines gracefully
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------
# Defaults: seed dictionary if empty
# ---------------------------

def seed_dictionary_if_needed():
    if DICT_PATH.exists():
        return
    ensure_data_dir()
    defaults = [
        {"id": "dict_water_001", "text": "In Islamic dream interpretation, water symbolizes knowledge, faith, and purification. Clear water reflects righteousness and blessings, while murky water warns of trials or confusion."},
        {"id": "dict_falling_001", "text": "Falling in a dream may signify loss of status, weakness in faith, or a warning to correct one’s deeds before harm befalls."},
        {"id": "dict_teeth_001", "text": "Teeth can relate to family members or personal strength. Losing teeth may reflect anxiety about status or relatives’ wellbeing."},
        {"id": "dict_snake_001", "text": "A snake can symbolize an enemy, hidden fear, or transformative energy; the outcome depends on whether you defeat or befriend it."},
        {"id": "dict_flight_001", "text": "Flying often points to ambition, spiritual elevation, or escape from constraints; controlled flight is positive, chaotic flight suggests instability."},
    ]
    write_jsonl(DICT_PATH, defaults)


# ---------------------------
# Session state bootstrap
# ---------------------------

def init_state():
    if "dict_df" not in st.session_state:
        seed_dictionary_if_needed()
        dict_rows = read_jsonl(DICT_PATH)
        st.session_state.dict_df = pd.DataFrame(dict_rows)
    if "history" not in st.session_state:
        st.session_state.history = read_jsonl(HISTORY_PATH)
    if "mode" not in st.session_state:
        st.session_state.mode = "Balanced"


# ---------------------------
# Interpretation engine (simple rule-based starter)
# ---------------------------

KEYWORD_HINTS = {
    # keyword: id in dictionary
    "water": "dict_water_001",
    "river": "dict_water_001",
    "ocean": "dict_water_001",
    "falling": "dict_falling_001",
    "teeth": "dict_teeth_001",
    "snake": "dict_snake_001",
    "flying": "dict_flight_001",
    "flight": "dict_flight_001",
}


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def match_entries(dream_text: str, df: pd.DataFrame) -> list[dict]:
    """Return best-matching dictionary entries by simple keyword scan + fuzzy-ish fallback.
    This is intentionally simple; you can swap it with an ML model later.
    """
    text = normalize_text(dream_text)

    hits: list[dict] = []
    # 1) direct keyword → id mapping
    for kw, entry_id in KEYWORD_HINTS.items():
        if re.search(rf"\b{re.escape(kw)}\b", text):
            row = df[df["id"] == entry_id]
            if not row.empty:
                hits.append(row.iloc[0].to_dict())

    # 2) naive token overlap (fallback)
    tokens = set(re.findall(r"[a-zA-Z]+", text))
    for _, row in df.iterrows():
        t = set(re.findall(r"[a-zA-Z]+", row.get("text", "").lower()))
        if not t:
            continue
        score = len(tokens & t)
        if score >= 4 and row.to_dict() not in hits:
            hits.append(row.to_dict())

    return hits[:5]


def craft_interpretation(dream_text: str, mode: str, matches: list[dict]) -> str:
    # Headline
    header = f"**Reading ({mode})**\n\n"

    # Evidence from matches
    bullets = []
    for m in matches:
        bullets.append(f"• {m['text']}")

    if not bullets:
        bullets.append("• No direct symbol match found. Consider context: emotions, people present, colors, and outcomes.")

    # Mode flavoring
    if mode == "Islamic":
        flavor = (
            "\n\n**Islamic lens:** Focus on righteousness, purification, and warnings. Reflect on recent deeds, prayers, and intentions. "
            "Seek wisdom from scholars for complex or recurring dreams."
        )
    elif mode == "Psychology":
        flavor = (
            "\n\n**Psychology lens:** Consider stressors, relationships, and unmet needs. Recurring symbols may mirror persistent anxieties or goals."
        )
    else:
        flavor = (
            "\n\n**Balanced lens:** Synthesize spiritual meaning with personal context—how you felt during and after the dream matters."
        )

    guidance = (
        "\n\n**Actionable reflection:** Write a 3–5 line journal entry: (1) strongest emotion, (2) key symbol, (3) real-life parallel, (4) one small improvement for tomorrow."
    )

    return header + "\n".join(bullets) + flavor + guidance


# ---------------------------
# UI helpers
# ---------------------------

def spacer(h: int = 8):
    st.write("""
    <div style="height: %dpx"></div>
    """ % h, unsafe_allow_html=True)


def subheader(text: str):
    st.markdown(f"### {text}")


# ---------------------------
# Pages
# ---------------------------

def page_interpret():
    subheader("Interpret a Dream")
    default_demo = "I was flying over a calm ocean and then my teeth started to fall out while I looked for my family."
    dream = st.text_area("Describe your dream", value=default_demo, height=150, help="Be specific: symbols, colors, emotions, people, outcomes.")

    st.session_state.mode = st.radio("Interpretation mode", ["Balanced", "Islamic", "Psychology"], horizontal=True, index=["Balanced","Islamic","Psychology"].index(st.session_state.mode))

    if st.button("Generate Interpretation", use_container_width=True):
        matches = match_entries(dream, st.session_state.dict_df)
        out = craft_interpretation(dream, st.session_state.mode, matches)
        st.markdown(out)

        # Save to history
        st.session_state.history.append({
            "ts": datetime.utcnow().isoformat(),
            "mode": st.session_state.mode,
            "dream": dream,
            "matches": [m["id"] for m in matches],
            "output": out,
        })
        ensure_data_dir()
        write_jsonl(HISTORY_PATH, st.session_state.history)

    if st.session_state.history:
        with st.expander("Recent interpretations"):
            for item in reversed(st.session_state.history[-5:]):
                st.markdown(f"- *{item['mode']}* — {item['ts']}\n\n{item['output']}")


def page_dictionary():
    subheader("Dictionary (JSONL)")
    st.caption("Each row has an 'id' and a 'text' field. Use search to filter. Add your own entries below.")

    # Search/filter
    q = st.text_input("Search text", placeholder="e.g., snake, water, success…")
    df = st.session_state.dict_df
    if q:
        m = df["text"].str.contains(q, case=False, na=False) | df["id"].str.contains(q, case=False, na=False)
        view = df[m]
    else:
        view = df

    st.dataframe(view, use_container_width=True, hide_index=True)

    spacer(4)
    st.markdown("**Add / Edit Entry**")
    cid = st.text_input("ID", placeholder="dict_symbol_123")
    ctext = st.text_area("Meaning / Guidance", placeholder="Write a clear, concise interpretation.", height=120)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Add / Update", use_container_width=True):
            if cid and ctext:
                # upsert
                exists = st.session_state.dict_df["id"] == cid
                if exists.any():
                    st.session_state.dict_df.loc[exists, "text"] = ctext
                    st.success(f"Updated: {cid}")
                else:
                    st.session_state.dict_df = pd.concat([
                        st.session_state.dict_df,
                        pd.DataFrame([{"id": cid, "text": ctext}])
                    ], ignore_index=True)
                    st.success(f"Added: {cid}")
                ensure_data_dir()
                write_jsonl(DICT_PATH, st.session_state.dict_df.to_dict(orient="records"))
            else:
                st.error("Please provide both ID and text.")

    with col2:
        if st.button("Delete by ID", use_container_width=True):
            if cid:
                before = len(st.session_state.dict_df)
                st.session_state.dict_df = st.session_state.dict_df[st.session_state.dict_df["id"] != cid]
                after = len(st.session_state.dict_df)
                if after < before:
                    st.success(f"Deleted: {cid}")
                    ensure_data_dir()
                    write_jsonl(DICT_PATH, st.session_state.dict_df.to_dict(orient="records"))
                else:
                    st.warning("ID not found.")
            else:
                st.error("Enter an ID to delete.")

    with col3:
        if st.button("Export JSONL", use_container_width=True):
            st.session_state["export_ready"] = True

    if st.session_state.get("export_ready"):
        text = "\n".join(json.dumps(r, ensure_ascii=False) for r in st.session_state.dict_df.to_dict(orient="records"))
        st.download_button("Download dream_dict.jsonl", data=text, file_name="dream_dict.jsonl", mime="application/json")


def page_data_io():
    subheader("Import / Merge Dictionary")
    st.caption("Upload a JSONL file with entries {id, text}. We'll merge by ID (uploaded rows replace existing IDs).")
    up = st.file_uploader("Upload JSONL", type=["jsonl","txt","json"])
    if up:
        new_rows = []
        try:
            for raw in up.read().decode("utf-8").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                new_rows.append(json.loads(raw))
        except Exception as e:
            st.error(f"Could not parse file: {e}")
            return

        # merge by ID
        base = {r["id"]: r for r in st.session_state.dict_df.to_dict(orient="records")}
        for r in new_rows:
            if "id" in r and "text" in r:
                base[r["id"]] = {"id": r["id"], "text": r["text"]}
        merged = list(base.values())
        st.session_state.dict_df = pd.DataFrame(merged)
        ensure_data_dir()
        write_jsonl(DICT_PATH, merged)
        st.success(f"Merged {len(new_rows)} rows. Total entries: {len(merged)}")

    spacer(6)
    subheader("Interpretation History")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
        if st.button("Export History JSONL", use_container_width=True):
            text = "\n".join(json.dumps(r, ensure_ascii=False) for r in st.session_state.history)
            st.download_button("Download history.jsonl", data=text, file_name="history.jsonl", mime="application/json")
    else:
        st.info("No history yet. Generate an interpretation first.")


def page_analytics():
    subheader("Analytics (quick view)")

    if st.session_state.history:
        texts = " ".join([h.get("dream", "") for h in st.session_state.history]).lower()
        words = re.findall(r"[a-zA-Z]+", texts)
        common = Counter([w for w in words if len(w) >= 4]).most_common(12)
        if common:
            chart_df = pd.DataFrame(common, columns=["word", "count"]).set_index("word")
            st.bar_chart(chart_df)
        else:
            st.info("Not enough data for a chart yet.")
    else:
        st.info("No interpretations yet—come back after generating a few.")


# ---------------------------
# Main
# ---------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

    # Simple style tweaks
    st.markdown(
        """
        <style>
            .stButton>button {border-radius: 12px; padding: 0.6rem 1rem;}
            .block-container {max-width: 1100px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    init_state()

    st.title(APP_TITLE)
    st.caption("A clean, customizable interface to explore dream symbols and meanings.")

    page = st.sidebar.radio("Navigate", ["Interpret", "Dictionary", "Import/Export", "Analytics"], index=0)

    if page == "Interpret":
        page_interpret()
    elif page == "Dictionary":
        page_dictionary()
    elif page == "Import/Export":
        page_data_io()
    elif page == "Analytics":
        page_analytics()

    st.sidebar.markdown("---")
    st.sidebar.write("**Tips**")
    st.sidebar.caption("Add your own entries to the dictionary. Keep meanings short and clear. Use Analytics to spot common themes.")


if __name__ == "__main__":
    main()
