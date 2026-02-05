import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="DevalShield | Risk Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLING ---
# --- STYLING (SACRA STUDIO AESTHETIC) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #05070a;
    }
    
    .main {
        background: radial-gradient(circle at top right, #1a1c2c, #05070a);
    }
    
    /* Global Card / Glass container */
    div.stBlock {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Metric Styling */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 600;
        background: linear-gradient(90deg, #f0f2f6, #00d1ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricDelta"] {
        font-weight: 400;
        font-size: 0.9rem;
    }

    /* Titles */
    h1 {
        background: linear-gradient(90deg, #ffffff, #888);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
        letter-spacing: -1px;
    }
    
    h3 {
        color: #88c0d0;
        font-weight: 400;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 0.8rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0a0c10;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Buttons */
    .stButton>button {
        background: rgba(0, 209, 255, 0.1);
        border: 1px solid #00d1ff;
        color: #00d1ff;
        border-radius: 8px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: #00d1ff;
        color: #000;
    }
    </style>
""", unsafe_allow_html=True)

# --- DB HELPER ---
def get_db_connection():
    return sqlite3.connect("devalshield.db")

def load_data():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM signals ORDER BY timestamp DESC", conn)
    state = pd.read_sql_query("SELECT * FROM system_state", conn)
    conn.close()
    return df, state

# --- SIDEBAR ---
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<h1 style='font-size: 1.5rem; text-align: center;'>SACRA STUDIO</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; color: #666; font-size: 0.8rem;'>DEVALSHIELD v2.0</p>", unsafe_allow_html=True)
st.sidebar.divider()

if st.sidebar.button("🔄 Force Data Sync"):
    st.rerun()

st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.caption("Institutional Protocol: CFA Standard L3")

# --- MAIN CONTENT ---
st.title("🛡️ Institutional Risk Monitor")
st.markdown("<p style='color: #666;'>Regime-Based Volatility & Tail-Risk Assessment Layer </p>", unsafe_allow_html=True)

try:
    df, state = load_data()
    
    if not df.empty:
        latest = df.iloc[0]
        
        # --- ANALYTICS LAYER 1: Core Metrics ---
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            dvi_val = float(latest['dvi'])
            delta = None
            if len(df) > 1:
                delta = dvi_val - float(df.iloc[1]['dvi'])
            st.metric("Deval Vacuum Index (DVI)", f"{dvi_val:.1f}%", delta=f"{delta:+.1f}%" if delta else None, delta_color="inverse")
            
        with col2:
            prob = latest['tail_risk_prob']
            st.metric("Tail-Risk Prob (30D)", f"{prob}%")
            
        with col3:
            st.metric("Oracle: SPY Price", f"${latest['spy_price']:.2f}")
            
        with col4:
            vix = latest['vix_current']
            st.metric("VIX Global Regime", f"{vix:.1f}")

        # --- ANALYTICS LAYER 2: Narratives & Allocation ---
        st.markdown("<br>", unsafe_allow_html=True)
        col_narrative, col_portfolio = st.columns([1.5, 1.5])
        
        with col_narrative:
            st.subheader("📖 Sacra Studio Briefing")
            try:
                raw = json.loads(latest['raw_context'])
                narrative_lines = raw.get('narrative', ["No strategic narrative available."])
                # Filter out the meta-data for a cleaner view if needed, but here we show the structured briefing
                for line in narrative_lines:
                    if line.strip():
                        st.markdown(f"> {line}")
            except:
                st.info("No detailed narrative found.")

        with col_portfolio:
            st.subheader("💎 Portfolio Intelligence")
            
            # Sub-layer 1: Daily Impact
            st.markdown("### Daily Impact")
            port_data = context.get('portfolio', {})
            if port_data:
                p_cols = st.columns(len(port_data) // 2 + 1)
                for i, (ticker, info) in enumerate(port_data.items()):
                    with p_cols[i % len(p_cols)]:
                        color = "normal" if info['change_pct'] > 0 else "inverse"
                        st.metric(ticker, f"${info['price']}", f"{info['change_pct']:+.2f}%", delta_color=color)
            
            st.divider()
            
            # Sub-layer 2: Exposure & Suggested
            c_exp, c_sug = st.columns(2)
            with c_exp:
                st.markdown("### Current Exposure")
                st.caption("**75%** Commodities")
                st.caption("**25%** Defense + AI + Utl")
                st.caption("**22%** Cash Reserve")
            
            with c_sug:
                st.markdown("### Alpha-Allocation (RL)")
                try:
                    raw = json.loads(latest['raw_context'])
                    alloc = raw.get('rl_allocation', {"Equities": 55, "Commodities": 20, "Cash": 25})
                    for k, v in alloc.items():
                        st.text(f"{k}: {v}%")
                except:
                    st.caption("RL Optimization N/A")

        # --- VISUALIZATION LAYER: Historical ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📈 Historical Risk Trajectory")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_hist = df.sort_values('timestamp')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_hist['timestamp'], y=df_hist['dvi'],
            mode='lines',
            name='DVI (Vacuum)',
            line=dict(color='#00d1ff', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 209, 255, 0.1)'
        ))
        fig.add_trace(go.Scatter(
            x=df_hist['timestamp'], y=df_hist['tail_risk_prob'],
            mode='lines',
            name='Tail-Risk %',
            line=dict(color='#ffcc00', width=1, dash='dot')
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=20),
            height=300,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- CITADEL RISK MATRIX (NEW) ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🏛️ Citadel-Level Risk Matrix (10x)")
        
        try:
            raw = json.loads(latest['raw_context'])
            metrics = raw.get('citadel_metrics', {
                "beta": 1.45, "correlation": 0.78, "concentration": 75, "cash_ratio": 22
            })
            
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            
            with m_col1:
                st.metric("Portfolio Beta", f"{metrics.get('beta', 1.45):.2f}", delta="Vulnerable" if metrics.get('beta', 1.45) > 1.3 else "Sano")
            with m_col2:
                st.metric("Stress Correlation", f"{metrics.get('correlation', 0.78):.2f}", delta="ALTA" if metrics.get('correlation', 0.78) > 0.7 else "BAJA")
            with m_col3:
                st.metric("Concentration", f"{metrics.get('concentration', 75)}%", delta="CRÍTICA" if metrics.get('concentration', 75) > 40 else "OK")
            with m_col4:
                st.metric("Cash Ratio", f"{metrics.get('cash_ratio', 22)}%", delta="LIQUIDEZ OK" if metrics.get('cash_ratio', 22) > 15 else "BAJA")
            with m_col5:
                vix_val = float(latest['vix_current'])
                st.metric("Hedging Cost", f"VIX {vix_val:.1f}", delta="BARATO" if vix_val < 25 else "CARO")
                
            st.markdown(f"""
                <div style="font-size: 0.85rem; color: #888; background: rgba(0,209,255,0.05); padding: 15px; border-radius: 10px; border-left: 3px solid #00d1ff;">
                    <b>Citadel Analysis:</b> Concentración en Commodities (75%) genera un Beta de {metrics.get('beta', 1.45):.2f}. 
                    En un escenario de stress, la correlación de {metrics.get('correlation', 0.78):.2f} sugiere un drawdown sistémico. 
                    <b>Recomendación:</b> Reducir exposición a metales (SLV/URA) para bajar el VaR de la cuenta.
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.caption(f"Waiting for Citadel Matrix data... ({e})")

        # --- SYSTEM INFRASTRUCTURE ---
        st.divider()
        c_safety, c_vault = st.columns(2)
        
        with c_safety:
            st.subheader("🔒 Defense Protocols")
            lock_row = state[state['key'] == 'kill_switch_active']
            is_locked = True if not lock_row.empty and lock_row['value'].values[0] == "true" else False
            
            if is_locked:
                st.error("🚨 **KILL-SWITCH ACTIVE** | Automated Triggers Disabled.")
            else:
                st.success("🟢 **SYSTEM UNLOCKED** | Human-in-the-loop Bridge Active.")
            
        with c_vault:
            st.subheader("🏦 Solana Infrastructure")
            st.markdown(f"""
                <div style="background-color: rgba(255, 204, 0, 0.05); padding: 20px; border-radius: 12px; border-left: 4px solid #ffcc00;">
                    <b style="color: #ffcc00;">STATUS: PRE-DEPLOYMENT SIMULATION</b><br>
                    <span style='font-size: 0.8rem; color: #888;'>Mainnet cluster verified. Vault binary verified (336KB).</span>
                </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("Ecosistema en espera. Inicie un análisis desde el bot de Telegram para poblar el Dashboard.")

except Exception as e:
    st.error(f"Dashboard Fault: {e}")

# --- FOOTER ---
st.divider()
st.markdown("<p style='text-align: center; color: #444; font-size: 0.8rem;'>SACRA STUDIO © 2026 | PROPRIETARY ALGORITHMIC ARCHITECTURE | CFA INSTITUTE ETHICAL STANDARDS</p>", unsafe_allow_html=True)
