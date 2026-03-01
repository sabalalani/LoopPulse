import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# PAGE CONFIG & THEME
# ============================================================================
st.set_page_config(page_title="Loop Pulse — Stakeholder Insights", layout="wide")

COLORS = {
    "blue": "#3b82f6", "cyan": "#06b6d4", "green": "#10b981",
    "orange": "#f59e0b", "purple": "#8b5cf6", "red": "#ef4444",
    "bg": "#0a0e17", "surface": "#111827", "text": "#e2e8f0", "muted": "#94a3b8"
}

st.markdown(f"""
<style>
    .stApp {{ background-color: {COLORS['bg']}; color: {COLORS['text']}; }}
    .stakeholder-card {{
        background-color: {COLORS['surface']};
        border: 1px solid #1e293b;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
    }}
    .metric-label {{ color: {COLORS['muted']}; font-size: 14px; text-transform: uppercase; }}
    .metric-value {{ color: {COLORS['text']}; font-size: 24px; font-weight: bold; }}
    .recommendation-box {{
        background-color: #1e293b;
        border-left: 5px solid {COLORS['blue']};
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("loop_pulse_features2.csv")
    df['year_month_dt'] = pd.to_datetime(df['year_month'])
    return df

df = load_data()

# ============================================================================
# SIDEBAR - DYNAMIC FILTERS
# ============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2102/2102633.png", width=80) 
    st.title("User Portal")
    
    role = st.selectbox(
        "Perspective:",
        ["City Official / Alderman", "Business Owner", "Real Estate Developer"]
    )
    
    st.divider()
    st.subheader("🛠️ Global Filters")
    
    # 1. Date Range Filter
    min_date = df['year_month_dt'].min().date()
    max_date = df['year_month_dt'].max().date()
    date_range = st.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    # 2. Transit Proximity Filter
    #max_dist = int(df['dist_to_nearest_cta'].max())
    #transit_limit = st.slider("Max Distance to CTA (m)", 0, max_dist, max_dist)
    
    # 3. Crime Volume Filter
    #crime_limit = st.slider("Max Monthly Crimes", 0, int(df['total_crimes'].max()), int(df['total_crimes'].max()))

    # Apply Filters to the Dataframe
    if len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered_df = df[(df['year_month_dt'] >= start_date) & (df['year_month_dt'] <= end_date)]
    else:
        filtered_df = df.copy()
    
    #filtered_df = filtered_df[filtered_df['dist_to_nearest_cta'] <= transit_limit]
    #filtered_df = filtered_df[filtered_df['total_crimes'] <= crime_limit]
    
    # Block Selection (Filtered by the sliders above)
    available_blocks = sorted(filtered_df['block_id'].unique())
    if not available_blocks:
        st.error("No blocks match these filters. Please adjust.")
        st.stop()
        
    selected_block = st.selectbox("Focus Area (Block ID):", available_blocks)
    
    # Data specific to the selected block and date range
    block_history = filtered_df[filtered_df['block_id'] == selected_block].sort_values('year_month_dt')
    block_data = block_history.iloc[-1] 
    loop_avg_arrest = filtered_df['arrest_rate'].mean()

# ============================================================================
# HEADER
# ============================================================================
st.title(f"{role} View")
st.markdown(f"Tailored intelligence for **{selected_block}** | Analyzing {len(block_history)} months of filtered data")

# ============================================================================
# ROLE 1: CITY OFFICIAL / ALDERMAN
# ============================================================================
if role == "City Official / Alderman":
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Policy Performance")
        st.markdown(f"""
        <div class="stakeholder-card">
            <div class="metric-label">Public Safety Efficiency</div>
            <div class="metric-value">{block_data['arrest_rate']*100:.1f}% Arrest Rate</div>
            <br>
            <div class="metric-label">Infrastructure Load</div>
            <div class="metric-value">{int(block_data['total_311_requests'])} Monthly Requests</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("**Policy Recommendation:**")
        st.write(f"Focus: Block {selected_block} is showing a {block_history['total_311_requests'].corr(block_history['total_crimes']):.2f} correlation between infrastructure reports and safety incidents.")

    with col2:
        st.subheader("Infrastructure vs. Safety (Scatter)")
        fig1 = px.scatter(block_history, x="total_311_requests", y="total_crimes", trendline="ols", color_discrete_sequence=[COLORS['orange']])
        fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", height=300)
        st.plotly_chart(fig1, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Historical Crime Volatility")
        fig2 = px.line(block_history, x="year_month_dt", y="total_crimes", color_discrete_sequence=[COLORS['red']])
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", height=300)
        st.plotly_chart(fig2, use_container_width=True)
    with col4:
        st.subheader("Arrest Rate Benchmark")
        fig3 = go.Figure(go.Bar(x=['This Block', 'Loop Avg'], y=[block_data['arrest_rate'], loop_avg_arrest], marker_color=[COLORS['blue'], COLORS['muted']]))
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", height=300)
        st.plotly_chart(fig3, use_container_width=True)

# ============================================================================
# ROLE 2: BUSINESS OWNER
# ============================================================================
elif role == "Business Owner":
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Environment Radar")
        fig1 = go.Figure(go.Scatterpolar(r=[block_data['active_business_count'], (1 - block_data['night_crime_ratio']) * 100, 
               block_data['business_diversity_index'] * 100, block_data['business_health_score']],
            theta=['Density','Safety','Diversity','Health'], fill='toself', line_color=COLORS['cyan']))
        fig1.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), paper_bgcolor='rgba(0,0,0,0)', font_color="white", height=350)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Action Center")
        st.markdown(f"""
        <div class="stakeholder-card">
            <div class="metric-label">Competitor Density</div>
            <div class="metric-value">{int(block_data['active_business_count'])} Neighbors</div>
            <br>
            <div class="metric-label">Current BHS</div>
            <div class="metric-value">{block_data['business_health_score']:.1f} Score</div>
        </div>
        """, unsafe_allow_html=True)
        st.success("**Retail Tip:**")
        st.write("Cross-shopping potential is linked to your Diversity Index. Focus on local partnerships.")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("12-Month Health Stability")
        fig2 = px.area(block_history.tail(12), x="year_month_dt", y="business_health_score", color_discrete_sequence=[COLORS['green']])
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", height=300)
        st.plotly_chart(fig2, use_container_width=True)
    with col4:
        st.subheader("Primary Crime Drivers")
        crime_mix = block_data[['theft_count', 'violent_crime_count', 'property_crime_count']]
        fig3 = px.bar(crime_mix, color_discrete_sequence=[COLORS['red']])
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", height=300, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

# ============================================================================
# ROLE 3: REAL ESTATE DEVELOPER
# ============================================================================
elif role == "Real Estate Developer":
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Investment Growth Curve")
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig1.add_trace(go.Scatter(x=block_history['year_month_dt'], y=block_history['business_health_score'], name="Health", line=dict(color=COLORS['green'])), secondary_y=False)
        fig1.add_trace(go.Bar(x=block_history['year_month_dt'], y=block_history['net_business_change'], name="Growth", marker_color=COLORS['blue'], opacity=0.4), secondary_y=True)
        fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", height=300, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.subheader("Yield Matrix (Health vs Crime)")
        fig2 = px.scatter(block_history, x="crimes_per_business", y="business_health_score", size="active_business_count", color="business_health_score", color_continuous_scale="RdYlGn")
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", height=300, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Transit Ridership Trends")
        fig3 = px.line(block_history, x="year_month_dt", y="cta_total_loop_ridership", color_discrete_sequence=[COLORS['cyan']])
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", height=300)
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        st.subheader("Development Context")
        st.markdown(f"""
        <div class="stakeholder-card">
            <div class="metric-label">CTA Proximity</div>
            <div class="metric-value">{int(block_data['dist_to_nearest_cta'])} Meters</div>
            <br>
            <div class="metric-label">Diversity Trend</div>
            <div class="metric-value">{block_data['business_diversity_index']:.2f} Index</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
#st.divider()
#st.caption("Each view adapts to the global filters selected in the sidebar.")