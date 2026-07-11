"""ShopFloor AI Dashboard — KPIs, Failure Predictor, RAG Chat, Drift Monitor."""

import os, httpx
import pandas as pd
import plotly.express as px
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8001")

st.set_page_config(page_title="ShopFloor AI", page_icon="🏭", layout="wide")
st.sidebar.title("🏭 ShopFloor AI")
page = st.sidebar.radio("Navigate", ["📊 KPI Dashboard", "🔮 Failure Predictor", "💬 Ask ShopFloor AI", "📈 Drift Monitor"])


def load_data():
    path = "data/raw/ai4i2020.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    st.error("Dataset not found. Run `make setup` first.")
    return None


if page == "📊 KPI Dashboard":
    st.title("📊 Manufacturing KPI Dashboard")
    df = load_data()
    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Units", f"{len(df):,}")
        c2.metric("Failure Rate", f"{df['Machine failure'].mean():.2%}")
        c3.metric("Avg Torque", f"{df['Torque [Nm]'].mean():.1f} Nm")
        c4.metric("Avg Tool Wear", f"{df['Tool wear [min]'].mean():.0f} min")

        st.divider()
        col_l, col_r = st.columns(2)
        with col_l:
            mode_counts = {m: int(df[m].sum()) for m in ["TWF", "HDF", "PWF", "OSF", "RNF"]}
            fig = px.bar(x=list(mode_counts.keys()), y=list(mode_counts.values()),
                         title="Failure Mode Distribution", labels={"x": "Mode", "y": "Count"},
                         color=list(mode_counts.values()), color_continuous_scale=["#22c55e", "#ef4444"])
            fig.update_layout(template="plotly_dark", height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            by_type = df.groupby("Type")["Machine failure"].mean().reset_index()
            fig = px.bar(by_type, x="Type", y="Machine failure", title="Failure Rate by Product Type",
                         color="Machine failure", color_continuous_scale=["#22c55e", "#ef4444"])
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.scatter(df, x="Rotational speed [rpm]", y="Torque [Nm]",
                             color=df["Machine failure"].map({0: "Pass", 1: "Fail"}),
                             title="Speed vs Torque (colored by failure)",
                             color_discrete_map={"Pass": "#22c55e", "Fail": "#ef4444"}, opacity=0.5)
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig = px.histogram(df, x="Tool wear [min]", color=df["Machine failure"].map({0: "Pass", 1: "Fail"}),
                               nbins=50, title="Tool Wear Distribution",
                               color_discrete_map={"Pass": "#22c55e", "Fail": "#ef4444"})
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)


elif page == "🔮 Failure Predictor":
    st.title("🔮 Machine Failure Predictor")
    c1, c2 = st.columns(2)
    with c1:
        air_t = st.slider("Air temperature (K)", 295.0, 305.0, 300.0, 0.1)
        proc_t = st.slider("Process temperature (K)", 305.0, 315.0, 310.0, 0.1)
        rpm = st.slider("Rotational speed (rpm)", 1000, 2800, 1500, 10)
    with c2:
        torque = st.slider("Torque (Nm)", 10.0, 80.0, 40.0, 0.5)
        wear = st.slider("Tool wear (min)", 0, 250, 100, 1)
        ptype = st.selectbox("Product type", ["L", "M", "H"])

    if st.button("🔮 Predict", type="primary", use_container_width=True):
        payload = {"Air temperature [K]": air_t, "Process temperature [K]": proc_t,
                   "Rotational speed [rpm]": rpm, "Torque [Nm]": torque,
                   "Tool wear [min]": wear, "Type": ptype}
        try:
            r = httpx.post(f"{API_URL}/predict", json=payload, timeout=10)
            if r.status_code == 200:
                res = r.json()
                if res["failure_prediction"]:
                    st.error(f"❌ FAILURE PREDICTED — probability: {res['failure_probability']:.1%}")
                else:
                    st.success(f"✅ PASS — failure probability: {res['failure_probability']:.1%}")
                if res["risk_factors"]:
                    st.warning("⚠️ Risk factors:\n" + "\n".join(f"- {r}" for r in res["risk_factors"]))
        except httpx.ConnectError:
            st.info("API not running. Start with `make api`")


elif page == "💬 Ask ShopFloor AI":
    st.title("💬 Ask ShopFloor AI")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Ask about failure modes, tool wear, specs..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                r = httpx.post(f"{API_URL}/ask", json={"question": prompt}, timeout=30)
                if r.status_code == 200:
                    res = r.json()
                    st.markdown(res["answer"])
                    with st.expander(f"📚 Sources ({res['n_sources']})"):
                        for s in res["sources"]:
                            st.markdown(f"**{s['title']}** — {s.get('excerpt', '')[:150]}...")
                    st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
            except httpx.ConnectError:
                st.info("API not running")

    st.divider()
    st.caption("Quick questions:")
    for q in ["What causes HDF?", "When to replace the tool?", "What is OEE target?", "Power failure thresholds?"]:
        if st.button(q, key=q):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()


elif page == "📈 Drift Monitor":
    st.title("📈 Sensor Drift Monitor")
    df = load_data()
    if df is not None:
        df["index"] = range(len(df))
        c1, c2 = st.columns(2)
        with c1:
            roll = df["Torque [Nm]"].rolling(200).mean()
            fig = px.line(x=df["index"], y=roll, title="Torque Rolling Average (200-unit window)",
                          labels={"x": "Unit", "y": "Torque [Nm]"})
            fig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            roll = df["Tool wear [min]"].rolling(200).mean()
            fig = px.line(x=df["index"], y=roll, title="Tool Wear Rolling Average",
                          labels={"x": "Unit", "y": "Wear [min]"})
            fig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Upward trends signal data drift — model retraining may be needed.")
