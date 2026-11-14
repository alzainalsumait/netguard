"""NetGuard dashboard (Streamlit). Reads netguard.db written by detector.py.

Run from the src folder, venv active:
    streamlit run dashboard.py
Keep `python detector.py --live ...` running in another window to see live data.
"""
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH = Path(__file__).resolve().parent.parent / "netguard.db"

st.set_page_config(page_title="NetGuard", page_icon=None, layout="wide")


@st.cache_data(ttl=3)
def load(table, limit=50000):
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table} ORDER BY id DESC LIMIT {int(limit)}", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    if not df.empty:
        df["detected_at"] = pd.to_datetime(df["detected_at"], errors="coerce")
    return df


st.title("NetGuard - Network Intrusion Detection")
st.caption(f"Database: {DB_PATH}")

flows, incidents = load("flows"), load("incidents")
if flows.empty:
    st.info("No data yet. Run `python detector.py --live --seconds 90` first, then refresh.")
    st.stop()

# ---------- sidebar ----------
st.sidebar.header("Filters")
auto = st.sidebar.checkbox("Auto-refresh every 5 s", value=False)
min_conf = st.sidebar.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
types = sorted(incidents["attack_type"].unique()) if not incidents.empty else []
chosen = st.sidebar.multiselect("Attack types", types, default=types)
if st.sidebar.button("Refresh now"):
    load.clear()
    st.rerun()

inc = incidents[(incidents["confidence"] >= min_conf) & (incidents["attack_type"].isin(chosen))] \
    if not incidents.empty else incidents

# ---------- headline numbers ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Flows analysed", f"{len(flows):,}")
c2.metric("Alerts raised", f"{int(flows['is_alert'].sum()):,}")
c3.metric("Alert rate", f"{flows['is_alert'].mean() * 100:.1f}%")
c4.metric("Distinct attacker IPs", inc["src_ip"].nunique() if not inc.empty else 0)

tab_live, tab_hist, tab_log = st.tabs(["Live monitor", "Analytics", "Incident log"])

with tab_live:
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Traffic per 10 seconds")
        t = flows.dropna(subset=["detected_at"]).copy()
        t["bucket"] = t["detected_at"].dt.floor("10s")
        t["kind"] = t["is_alert"].map({1: "Alert", 0: "Normal"})
        series = t.groupby(["bucket", "kind"]).size().unstack(fill_value=0)
        st.line_chart(series)
    with right:
        st.subheader("Classification mix")
        st.bar_chart(flows["label"].value_counts())
    st.subheader("Most recent alerts")
    if inc.empty:
        st.success("No alerts match the current filters.")
    else:
        st.dataframe(inc.head(20)[["detected_at", "attack_type", "src_ip", "dst_ip",
                                   "dst_port", "confidence"]],
                     hide_index=True)

with tab_hist:
    if inc.empty:
        st.info("No incidents to analyse yet.")
    else:
        a, b = st.columns(2)
        with a:
            st.subheader("Incidents by attack type")
            st.bar_chart(inc["attack_type"].value_counts())
        with b:
            st.subheader("Top source IPs")
            st.bar_chart(inc["src_ip"].value_counts().head(10))
        st.subheader("Most targeted destination ports")
        st.bar_chart(inc["dst_port"].value_counts().head(15))
        st.subheader("Confidence distribution")
        st.bar_chart(pd.cut(inc["confidence"], bins=[0, .7, .8, .9, .95, 1.0]).value_counts().sort_index()
                     .rename(index=str))

with tab_log:
    st.subheader("Incident log")
    if inc.empty:
        st.info("Nothing logged for the current filters.")
    else:
        st.dataframe(inc, hide_index=True)
        st.download_button("Download as CSV", inc.to_csv(index=False).encode("utf-8"),
                           file_name="netguard_incidents.csv", mime="text/csv")

if auto:
    import time
    time.sleep(5)
    load.clear()
    st.rerun()
