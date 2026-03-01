import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Loop Pulse — Economic Safety Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# FONT CONFIGURATION
# ============================================================================

FONT_FAMILY = '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
FONT_SIZES = {
    "title": 48,
    "subtitle": 24,
    "heading": 20,
    "body": 16,
    "small": 14,
    "tiny": 12,
    "metric": 36,
    "caption": 14
}

# ============================================================================
# COLOR SCHEME
# ============================================================================

COLORS = {
    "blue": "#3b82f6",
    "cyan": "#06b6d4",
    "green": "#10b981",
    "orange": "#f59e0b",
    "purple": "#8b5cf6",
    "red": "#ef4444",
    "pink": "#ec4899",
    "bg": "#0a0e17",
    "surface": "#111827",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
    "border": "#1e293b",
}

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {{
        background: linear-gradient(135deg, {COLORS['bg']} 0%, #0f172a 100%);
        font-family: {FONT_FAMILY};
    }}
    
    /* Hero section */
    .hero-title {{
        font-size: {FONT_SIZES['title']}px;
        font-weight: 700;
        background: linear-gradient(135deg, {COLORS['text']} 0%, {COLORS['blue']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }}
    
    .hero-subtitle {{
        font-size: {FONT_SIZES['subtitle']}px;
        color: {COLORS['muted']};
        font-weight: 400;
        margin-bottom: 2rem;
    }}
    
    /* Feature cards */
    .feature-card {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 20px;
        padding: 2rem;
        height: 100%;
        transition: transform 0.3s, border-color 0.3s;
    }}
    
    .feature-card:hover {{
        transform: translateY(-5px);
        border-color: {COLORS['blue']};
        box-shadow: 0 10px 30px -10px rgba(59, 130, 246, 0.3);
    }}
    
    .feature-icon {{
        font-size: 48px;
        margin-bottom: 1rem;
    }}
    
    .feature-title {{
        font-size: {FONT_SIZES['heading']}px;
        font-weight: 600;
        color: {COLORS['text']};
        margin-bottom: 1rem;
    }}
    
    .feature-description {{
        color: {COLORS['muted']};
        font-size: {FONT_SIZES['body']}px;
        line-height: 1.6;
    }}
    
    /* Stats section */
    .stat-card {{
        background: linear-gradient(135deg, {COLORS['surface']} 0%, #1a2234 100%);
        border: 1px solid {COLORS['border']};
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }}
    
    .stat-value {{
        font-size: 32px;
        font-weight: 700;
        color: {COLORS['blue']};
    }}
    
    .stat-label {{
        color: {COLORS['muted']};
        font-size: {FONT_SIZES['small']}px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    /* Navigation hint */
    .nav-hint {{
        background: linear-gradient(135deg, {COLORS['blue']}20, {COLORS['purple']}20);
        border: 1px solid {COLORS['blue']}40;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin: 2rem 0;
    }}
    
    .nav-hint-arrow {{
        font-size: 48px;
        color: {COLORS['blue']};
        animation: bounce-left 2s infinite;
    }}
    
    @keyframes bounce-left {{
        0%, 100% {{ transform: translateX(0); }}
        50% {{ transform: translateX(-15px); }}
    }}
    
    .custom-divider {{
        background: linear-gradient(90deg, transparent, {COLORS['border']}, transparent);
        height: 1px;
        margin: 3rem 0;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA FOR STATS
# ============================================================================

@st.cache_data
def load_stats():
    try:
        df = pd.read_csv("loop_pulse_features2.csv")
        return {
            'records': len(df),
            'blocks': df['block_id'].nunique(),
            'years': df['year_month'].str[:4].nunique(),
            'avg_bhs': df['business_health_score'].mean()
        }
    except:
        return {'records': 25000, 'blocks': 150, 'years': 23, 'avg_bhs': 35.2}

stats = load_stats()

# ============================================================================
# HERO SECTION
# ============================================================================

st.markdown('<div class="hero-title">Loop Pulse</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Economic Safety Intelligence Platform</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='stat-card'><div class='stat-value'>{stats['records']:,}</div><div class='stat-label'>Data Points</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='stat-card'><div class='stat-value'>{stats['blocks']}</div><div class='stat-label'>Blocks Analyzed</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='stat-card'><div class='stat-value'>{stats['years']}</div><div class='stat-label'>Years of Data</div></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='stat-card'><div class='stat-value'>{stats['avg_bhs']:.1f}</div><div class='stat-label'>Avg BHS</div></div>", unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ============================================================================
# EXECUTIVE SUMMARY
# ============================================================================

st.markdown("## Executive Summary")
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"""
    <div style='background-color: {COLORS['surface']}; padding: 2rem; border-radius: 20px; border: 1px solid {COLORS['border']};'>
        <p style='color: {COLORS['text']}; font-size: {FONT_SIZES['body']}px; line-height: 1.8;'>
        <strong style='color: {COLORS['blue']};'>Loop Pulse</strong> is an economic safety intelligence platform that connects the dots 
        between public safety data and economic vitality in Chicago's Loop. While existing tools map <em>where</em> crime happens, 
        Loop Pulse answers the critical follow-up question: <strong style='color: {COLORS['orange']};'>“So what?”</strong>
        </p>
        <p style='color: {COLORS['text']}; font-size: {FONT_SIZES['body']}px; line-height: 1.8; margin-top: 1rem;'>
        By correlating crime incident data with business performance metrics, foot traffic patterns, and 311 service requests, 
        Loop Pulse quantifies the economic ripple effects of safety issues. It empowers <strong>stakeholders</strong> to identify where 
        targeted interventions generate the highest return on investment.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    fig = go.Figure()
    points = {'Crime': [0.2, 0.3], 'Business': [0.5, 0.8], 'Traffic': [0.8, 0.3], 'ROI': [0.3, 0.5], 'Impact': [0.7, 0.5]}
    
    # Restored Connecting Lines
    for i in range(len(points)):
        keys = list(points.keys())
        for j in range(i+1, len(points)):
            p1, p2 = points[keys[i]], points[keys[j]]
            fig.add_shape(type="line", x0=p1[0], y0=p1[1], x1=p2[0], y1=p2[1], line=dict(color=COLORS['border'], width=1, dash="dot"))
    
    fig.add_trace(go.Scatter(
        x=[p[0] for p in points.values()], y=[p[1] for p in points.values()],
        mode='markers+text',
        marker=dict(size=[30, 40, 35, 25, 45], color=[COLORS['red'], COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['purple']]),
        text=list(points.keys()),
        textposition='top center',
        textfont=dict(color=COLORS['text'], size=12)
    ))
    
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, height=300, 
                      margin=dict(l=0, r=0, t=0, b=0), xaxis=dict(showgrid=False, showticklabels=False), yaxis=dict(showgrid=False, showticklabels=False))
    st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ============================================================================
# FEATURE SHOWCASE
# ============================================================================

st.markdown("## Our Solution")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class='feature-card'>
        <div class='feature-title'>Economic Impact Heatmap</div>
        <div class='feature-description'>
            Interactive block-level map overlaying crime incidents with business density. 
            Includes the <strong>Business Health Score (BHS)</strong> to track recovery vs decline.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='feature-card'>
        <div class='feature-title'>Intervention ROI Simulator</div>
        <div class='feature-description'>
            ML-powered what-if analysis. Model proposed changes (e.g., streetlights or grants) and 
            see projected economic uplift with 80% confidence intervals.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='feature-card'>
        <div class='feature-title'>Stakeholder Insights Portal</div>
        <div class='feature-description'>
            Tailored 4-dimension views for <strong>Aldermen, Business Owners, and Developers</strong>. 
            Translates complex ML data into specific policy, retail, and investment strategies.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ============================================================================
# NAVIGATION HINT
# ============================================================================

st.markdown(f"""
<div class='nav-hint'>
    <h3 style='color: {COLORS['text']}; margin-top: 1rem;'>Ready to explore?</h3>
    <p style='color: {COLORS['muted']}; font-size: {FONT_SIZES['body']}px; max-width: 600px; margin: 1rem auto;'>
        Use the <strong style='color: {COLORS['blue']};'>sidebar on the left</strong> to navigate:
    </p>
    <div style='display: flex; justify-content: center; gap: 1.5rem; margin-top: 1.5rem; flex-wrap: wrap;'>
        <div style='background-color: {COLORS['bg']}; padding: 0.75rem 1.5rem; border-radius: 50px; border: 1px solid {COLORS['border']}; color: {COLORS['text']};'>Economic Impact</div>
        <div style='background-color: {COLORS['bg']}; padding: 0.75rem 1.5rem; border-radius: 50px; border: 1px solid {COLORS['border']}; color: {COLORS['text']};'>ROI Simulator</div>
        <div style='background-color: {COLORS['bg']}; padding: 0.75rem 1.5rem; border-radius: 50px; border: 1px solid {COLORS['border']}; color: {COLORS['text']};'>Stakeholder Portal</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"<p style='color: {COLORS['muted']};'><strong>Data:</strong> Chicago Crime, Business Licenses, 311, CTA</p>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<p style='color: {COLORS['muted']};' align='right'><strong>Version:</strong> 2.5 | <strong>2026 DePaul University</strong></p>", unsafe_allow_html=True)