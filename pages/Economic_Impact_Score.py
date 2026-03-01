"""
Loop Pulse — Streamlit Dashboard
=================================
Interactive visualization of the Loop Pulse feature-engineered dataset.

Install:
    pip install streamlit plotly pandas numpy

Run:
    streamlit run streamlit_dashboard.py

Make sure loop_pulse_features2.csv is in the same folder.
"""

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
    page_title="Loop Pulse Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# FONT CONFIGURATION
# ============================================================================

FONT_FAMILY = '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
FONT_SIZES = {
    "title": 24,
    "subtitle": 18,
    "heading": 16,
    "body": 14,
    "small": 12,
    "tiny": 11,
    "metric": 32,
    "metric_label": 12,
    "caption": 13
}

# Custom CSS with font definitions
st.markdown(f"""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {{ 
        background-color: #0a0e17;
        font-family: {FONT_FAMILY};
    }}
    
    /* Global font settings */
    html, body, [class*="css"] {{
        font-family: {FONT_FAMILY};
    }}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{ 
        gap: 8px;
        font-family: {FONT_FAMILY};
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #111827;
        border-radius: 8px 8px 0 0;
        padding: 10px 24px;
        color: #94a3b8;
        font-family: {FONT_FAMILY};
        font-size: {FONT_SIZES["body"]}px;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #1e293b;
        color: #3b82f6;
        font-weight: 600;
    }}
    
    /* Metric cards */
    .metric-card {{
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-family: {FONT_FAMILY};
    }}
    .metric-value {{ 
        font-size: {FONT_SIZES["metric"]}px; 
        font-weight: 700; 
        color: #e2e8f0;
        font-family: {FONT_FAMILY};
        letter-spacing: -0.02em;
    }}
    .metric-label {{ 
        font-size: {FONT_SIZES["metric_label"]}px; 
        color: #94a3b8; 
        text-transform: uppercase; 
        letter-spacing: 1px;
        font-family: {FONT_FAMILY};
        font-weight: 500;
    }}
    
    /* Streamlit metric overrides */
    div[data-testid="stMetric"] {{
        background-color: #111827;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px;
        font-family: {FONT_FAMILY};
    }}
    div[data-testid="stMetric"] label {{
        font-family: {FONT_FAMILY};
        font-size: {FONT_SIZES["metric_label"]}px;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    div[data-testid="stMetric"] div {{
        font-family: {FONT_FAMILY};
        font-size: {FONT_SIZES["metric"]}px;
        font-weight: 700;
        color: #e2e8f0;
        letter-spacing: -0.02em;
    }}
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        font-family: {FONT_FAMILY};
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #e2e8f0;
    }}
    h1 {{ font-size: 32px; }}
    h2 {{ font-size: 28px; }}
    h3 {{ font-size: 24px; }}
    h4 {{ font-size: 20px; }}
    h5 {{ font-size: 16px; }}
    h6 {{ font-size: 14px; }}
    
    /* Paragraphs and text */
    p, li, span, div {{
        font-family: {FONT_FAMILY};
    }}
    
    /* Sidebar */
    .css-1d391kg, .css-1wrcr25 {{
        font-family: {FONT_FAMILY};
    }}
    
    /* Captions */
    .stCaption, caption {{
        font-family: {FONT_FAMILY};
        font-size: {FONT_SIZES["caption"]}px;
        color: #94a3b8;
        font-style: italic;
    }}
    
    /* DataFrames */
    .dataframe {{
        font-family: {FONT_FAMILY};
        font-size: {FONT_SIZES["small"]}px;
    }}
    
    /* Block container */
    .block-container {{ 
        padding-top: 2rem;
        font-family: {FONT_FAMILY};
    }}
    
    /* Sidebar text */
    .sidebar-content {{
        font-family: {FONT_FAMILY};
    }}
    
    /* Expanders */
    .streamlit-expanderHeader {{
        font-family: {FONT_FAMILY};
        font-weight: 500;
        font-size: {FONT_SIZES["body"]}px;
    }}
    
    /* Buttons */
    .stButton button {{
        font-family: {FONT_FAMILY};
        font-weight: 500;
        font-size: {FONT_SIZES["body"]}px;
    }}
    
    /* Select boxes and inputs */
    .stSelectbox, .stMultiSelect, .stSlider {{
        font-family: {FONT_FAMILY};
    }}
</style>
""", unsafe_allow_html=True)

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

PLOT_TEMPLATE = "plotly_dark"
# Base layout without title and legend to avoid conflicts
PLOT_LAYOUT = dict(
    paper_bgcolor="#111827",
    plot_bgcolor="#0a0e17",
    font=dict(
        family=FONT_FAMILY,
        color="#e2e8f0", 
        size=FONT_SIZES["body"]
    ),
    margin=dict(l=60, r=30, t=50, b=50),
    xaxis=dict(
        gridcolor="#1e293b",
        title_font=dict(family=FONT_FAMILY, size=FONT_SIZES["small"]),
        tickfont=dict(family=FONT_FAMILY, size=FONT_SIZES["tiny"])
    ),
    yaxis=dict(
        gridcolor="#1e293b",
        title_font=dict(family=FONT_FAMILY, size=FONT_SIZES["small"]),
        tickfont=dict(family=FONT_FAMILY, size=FONT_SIZES["tiny"])
    )
)

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data
def load_data():
    df = pd.read_csv(r"loop_pulse_features2.csv")
    df["year_month_dt"] = pd.to_datetime(df["year_month"])
    return df

try:
    df = load_data()
    # Ensure the new column exists. If running on old data, this prevents a crash.
    if "business_health_score" not in df.columns and "economic_impact_score" in df.columns:
        df.rename(columns={"economic_impact_score": "business_health_score"}, inplace=True)
except FileNotFoundError:
    st.error("❌ `loop_pulse_features2.csv` not found. Place it in the same folder as this script.")
    st.stop()

# ============================================================================
# PRECOMPUTE AGGREGATIONS
# ============================================================================

@st.cache_data
def compute_monthly(df):
    monthly = df.groupby("year_month").agg(
        total_crimes=("total_crimes", "sum"),
        avg_bhs=("business_health_score", "mean"),
        violent_crimes=("violent_crime_count", "sum"),
        theft_crimes=("theft_count", "sum"),
        property_crimes=("property_crime_count", "sum"),
        active_businesses=("active_business_count", "sum"),
        total_311=("total_311_requests", "sum"),
        cta_ridership=("cta_total_loop_ridership", "first"),
        streetlight_issues=("streetlight_issues", "sum"),
        graffiti=("graffiti_reports", "sum"),
        avg_arrest_rate=("arrest_rate", "mean"),
    ).reset_index()
    monthly["year_month_dt"] = pd.to_datetime(monthly["year_month"])
    return monthly.sort_values("year_month_dt")

@st.cache_data
def compute_top_blocks(df):
    blocks = df.groupby("block_id").agg(
        total_crimes=("total_crimes", "sum"),
        avg_bhs=("business_health_score", "mean"),
        lat=("lat_centroid", "first"),
        lon=("lon_centroid", "first"),
        avg_violent_ratio=("violent_to_total_ratio", "mean"),
        avg_businesses=("active_business_count", "mean"),
        avg_night_ratio=("night_crime_ratio", "mean"),
        avg_crimes_per_biz=("crimes_per_business", "mean")
    ).reset_index()
    
    return blocks.nlargest(30, "total_crimes")

@st.cache_data
def compute_correlations(df):
    key_features = [
        "total_crimes", "violent_crime_count", "theft_count", "arrest_rate",
        "night_crime_ratio", "active_business_count", "business_diversity_index",
        "total_311_requests", "infrastructure_issues", "streetlight_issues",
        "cta_total_loop_ridership", "crimes_per_business", "graffiti_reports",
        "net_business_change", "business_health_score"
    ]
    existing = [f for f in key_features if f in df.columns]
    corr = df[existing].corr()["business_health_score"].drop("business_health_score").sort_values()
    return corr

# ============================================================================
# SIDEBAR WITH ADVANCED FILTERS
# ============================================================================

with st.sidebar:
    st.markdown("### 🎯 Loop Pulse")
    st.markdown(
        f"<span style='color: {COLORS['muted']}; font-size: {FONT_SIZES['small']}px; "
        f"font-family: {FONT_FAMILY};'>Economic Safety Intelligence</span>", 
        unsafe_allow_html=True
    )
    st.divider()

    # ========================================================================
    # DATASET INFO (Always visible)
    # ========================================================================
    st.markdown(
        f"**📊 Dataset Snapshot**",
        help="Overview of the current dataset"
    )
    st.markdown(f"""
    - Records: **{len(df):,}**
    - Blocks: **{df['block_id'].nunique()}**
    - Period: **{df['year_month'].min()}** → **{df['year_month'].max()}**
    """)
    st.divider()
    
    # ========================================================================
    # TIME PERIOD FILTER
    # ========================================================================
    st.markdown("**📅 Time Period**")
    
    # Convert dates to datetime and then to date objects for the slider
    min_date = pd.to_datetime(df['year_month']).min().date()
    max_date = pd.to_datetime(df['year_month']).max().date()
    
    # Date range slider using date objects
    date_range = st.slider(
        "Select date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD",
        key="date_range"
    )
    
    # Convert back to datetime for filtering
    date_range_start = pd.to_datetime(date_range[0])
    date_range_end = pd.to_datetime(date_range[1])
    
    # Year selector (multi-select)
    available_years = sorted(df['year_month'].str[:4].unique())
    selected_years = st.multiselect(
        "Select years",
        options=available_years,
        default=available_years,
        key="year_filter"
    )
    
    st.divider()
    
    # ========================================================================
    # BLOCK FILTERS
    # ========================================================================
    st.markdown("**📍 Block Filters**")
    
    # Block selector (with search)
    all_blocks = sorted(df['block_id'].unique())
    selected_blocks = st.multiselect(
        "Select specific blocks",
        options=all_blocks,
        default=[],
        help="Leave empty to show all blocks",
        key="block_filter"
    )
    
    # Block type/category filters
    col1, col2 = st.columns(2)
    
    with col1:
        # Filter by business activity
        biz_threshold = st.slider(
            "Min active businesses",
            min_value=0,
            max_value=int(df['active_business_count'].max()),
            value=0,
            help="Show only blocks with at least this many businesses",
            key="biz_threshold"
        )
    
    with col2:
        # Filter by crime volume
        crime_threshold = st.slider(
            "Max crimes per month",
            min_value=0,
            max_value=int(df['total_crimes'].max()),
            value=int(df['total_crimes'].max()),
            help="Show only blocks with crimes below this threshold",
            key="crime_threshold"
        )
    
    st.divider()
    
    # ========================================================================
    # CRIME TYPE FILTERS
    # ========================================================================
    st.markdown("**🔫 Crime Filters**")
    
    # Crime severity filter
    crime_severity = st.select_slider(
        "Crime severity focus",
        options=["All Crimes", "Property Only", "Violent Only", "Quality of Life"],
        value="All Crimes",
        key="crime_severity"
    )
    
    # Time of day filter
    time_of_day = st.radio(
        "Time of day",
        options=["All Times", "Day (6am-6pm)", "Night (6pm-6am)", "Peak Hours (4pm-7PM)"],
        horizontal=True,
        key="time_of_day"
    )
    
    # Violent crime ratio threshold
    violent_threshold = st.slider(
        "Violent crime % threshold",
        min_value=0,
        max_value=100,
        value=100,
        help="Show only blocks where violent crime is above this %",
        key="violent_threshold"
    ) / 100
    
    st.divider()
    
    # ========================================================================
    # BUSINESS HEALTH FILTERS
    # ========================================================================
    st.markdown("**🏥 Business Health Filters**")
    
    # BHS range filter
    bhs_range = st.slider(
        "Business Health Score range",
        min_value=float(df['business_health_score'].min()),
        max_value=float(df['business_health_score'].max()),
        value=(float(df['business_health_score'].min()), 
               float(df['business_health_score'].max())),
        help="Filter blocks by their Business Health Score",
        key="bhs_range"
    )
    
    # Business trend filter
    biz_trend = st.selectbox(
        "Business trend",
        options=["All", "Growing (net_new > 0)", "Declining (net_new < 0)", "Stable"],
        index=0,
        key="biz_trend"
    )
    
    st.divider()
    
    # ========================================================================
    # INFRASTRUCTURE FILTERS
    # ========================================================================
    st.markdown("**🔧 Infrastructure Filters**")
    
    with st.expander("Infrastructure issues", expanded=False):
        min_311 = st.number_input("Min 311 requests", min_value=0, value=0, key="min_311")
        show_streetlight_issues = st.checkbox("Show only blocks with streetlight issues", key="streetlight")
        show_graffiti = st.checkbox("Show only blocks with graffiti reports", key="graffiti")
    
    st.divider()
    
    # ========================================================================
    # ADVANCED FILTERS (Collapsible)
    # ========================================================================
    with st.expander("⚙️ Advanced Filters", expanded=False):
        # Transit proximity
        transit_dist = st.slider(
            "Max distance to CTA (meters)",
            min_value=0,
            max_value=int(df['dist_to_nearest_cta'].max()),
            value=int(df['dist_to_nearest_cta'].max()),
            key="transit_dist"
        )
        
        # Arrest rate filter
        arrest_rate_min = st.slider(
            "Min arrest rate",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key="arrest_rate"
        )
        
        # Business diversity
        div_threshold = st.slider(
            "Min business diversity",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            key="div_threshold"
        )
    
    st.divider()
    
    # ========================================================================
    # DATA SOURCES (Collapsible)
    # ========================================================================
    with st.expander("📊 Data Sources", expanded=False):
        st.markdown("""
        - [Chicago Crime Data](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)
        - [Business Licenses](https://data.cityofchicago.org/Community-Economic-Development/Business-Licenses/r5kz-chrr)
        - [311 Service Requests](https://data.cityofchicago.org/Service-Requests/311-Service-Requests/v6vf-nfxy)
        - [CTA L Ridership](https://data.cityofchicago.org/Transportation/CTA-Ridership-L-Station-Entries-Daily-Totals/5neh-572f)
        """)
    
    st.divider()
    
    # ========================================================================
    # BHS FORMULA (Always visible)
    # ========================================================================
    st.markdown("**📐 BHS Formula Components**")
    st.markdown("""
    - **Economic Vitality**
      - Active business presence
      - License growth
    - **Public Safety Penalty**
      - Total crime volume
      - Violent crime severity
    - **Infrastructure Penalty**
      - 311 request volume
      - Resolution time
    """)
    
    st.divider()
    
    # ========================================================================
    # FILTER SUMMARY & RESET
    # ========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        # Count active filters
        active_filters = sum([
            date_range[0] > min_date or date_range[1] < max_date,
            len(selected_years) < len(available_years),
            len(selected_blocks) > 0,
            biz_threshold > 0,
            crime_threshold < df['total_crimes'].max(),
            crime_severity != "All Crimes",
            time_of_day != "All Times",
            violent_threshold < 1.0,
            bhs_range[0] > df['business_health_score'].min() or bhs_range[1] < df['business_health_score'].max(),
            biz_trend != "All",
            min_311 > 0,
            show_streetlight_issues,
            show_graffiti,
            transit_dist < df['dist_to_nearest_cta'].max(),
            arrest_rate_min > 0,
            div_threshold > 0
        ])
        st.caption(f"Active filters: {active_filters}")
    
    with col2:
        if st.button("🔄 Reset All Filters", key="reset"):
            # Clear all filter values by rerunning with cleared session state
            for key in list(st.session_state.keys()):
                if key not in ['df', 'monthly', 'top_blocks', 'correlations']:
                    del st.session_state[key]
            st.rerun()

# ============================================================================
# APPLY FILTERS TO DATA
# ============================================================================

def apply_filters(df, date_range_start, date_range_end, selected_years, selected_blocks, biz_threshold, 
                  crime_threshold, crime_severity, time_of_day, violent_threshold,
                  bhs_range, biz_trend, min_311, show_streetlight_issues, show_graffiti,
                  transit_dist, arrest_rate_min, div_threshold):
    """Apply all selected filters to the dataframe"""
    filtered_df = df.copy()
    
    # Convert year_month to datetime for filtering
    filtered_df['year_month_dt_filter'] = pd.to_datetime(filtered_df['year_month'])
    
    # Time period filter
    filtered_df = filtered_df[
        (filtered_df['year_month_dt_filter'] >= date_range_start) &
        (filtered_df['year_month_dt_filter'] <= date_range_end)
    ]
    
    # Year filter
    if selected_years:
        filtered_df = filtered_df[filtered_df['year_month'].str[:4].isin(selected_years)]
    
    # Block filter
    if selected_blocks:
        filtered_df = filtered_df[filtered_df['block_id'].isin(selected_blocks)]
    
    # Business count filter
    filtered_df = filtered_df[filtered_df['active_business_count'] >= biz_threshold]
    
    # Crime volume filter
    filtered_df = filtered_df[filtered_df['total_crimes'] <= crime_threshold]
    
    # Crime severity filter
    if crime_severity == "Property Only":
        filtered_df = filtered_df[filtered_df['property_crime_count'] > 0]
    elif crime_severity == "Violent Only":
        filtered_df = filtered_df[filtered_df['violent_crime_count'] > 0]
    elif crime_severity == "Quality of Life":
        quality_cols = ['graffiti_reports', 'streetlight_issues', 'business_complaints']
        filtered_df = filtered_df[filtered_df[quality_cols].sum(axis=1) > 0]
    
    # Time of day filter
    if time_of_day == "Day (6am-6pm)":
        filtered_df = filtered_df[filtered_df['night_crime_ratio'] < 0.5]
    elif time_of_day == "Night (6pm-6am)":
        filtered_df = filtered_df[filtered_df['night_crime_ratio'] >= 0.5]
    elif time_of_day == "Peak Hours (4pm-7PM)":
        filtered_df = filtered_df[filtered_df['peak_crime_hour'].between(16, 19)]  # 4-7 PM
    
    # Violent ratio filter
    filtered_df = filtered_df[filtered_df['violent_to_total_ratio'] <= violent_threshold]
    
    # BHS range filter
    filtered_df = filtered_df[
        (filtered_df['business_health_score'] >= bhs_range[0]) &
        (filtered_df['business_health_score'] <= bhs_range[1])
    ]
    
    # Business trend filter
    if biz_trend == "Growing (net_new > 0)":
        filtered_df = filtered_df[filtered_df['net_business_change'] > 0]
    elif biz_trend == "Declining (net_new < 0)":
        filtered_df = filtered_df[filtered_df['net_business_change'] < 0]
    elif biz_trend == "Stable":
        filtered_df = filtered_df[filtered_df['net_business_change'] == 0]
    
    # Infrastructure filters
    if min_311 > 0:
        filtered_df = filtered_df[filtered_df['total_311_requests'] >= min_311]
    
    if show_streetlight_issues:
        filtered_df = filtered_df[filtered_df['streetlight_issues'] > 0]
    
    if show_graffiti:
        filtered_df = filtered_df[filtered_df['graffiti_reports'] > 0]
    
    # Advanced filters
    filtered_df = filtered_df[filtered_df['dist_to_nearest_cta'] <= transit_dist]
    filtered_df = filtered_df[filtered_df['arrest_rate'] >= arrest_rate_min]
    filtered_df = filtered_df[filtered_df['business_diversity_index'] >= div_threshold]
    
    # Drop the temporary filter column
    filtered_df = filtered_df.drop(columns=['year_month_dt_filter'])
    
    return filtered_df

# Apply filters with corrected date parameters
filtered_df = apply_filters(
    df, 
    date_range_start, 
    date_range_end, 
    selected_years, 
    selected_blocks, 
    biz_threshold, 
    crime_threshold, 
    crime_severity, 
    time_of_day, 
    violent_threshold,
    bhs_range, 
    biz_trend, 
    min_311, 
    show_streetlight_issues, 
    show_graffiti,
    transit_dist, 
    arrest_rate_min, 
    div_threshold
)

# Recompute aggregations with filtered data
filtered_monthly = compute_monthly(filtered_df)
filtered_top_blocks = compute_top_blocks(filtered_df)
filtered_correlations = compute_correlations(filtered_df)

# Show filter summary in sidebar
if len(filtered_df) < len(df):
    st.sidebar.info(f"📊 Showing {len(filtered_df):,} of {len(df):,} records ({len(filtered_df)/len(df)*100:.1f}%)")

# ============================================================================
# HEADER
# ============================================================================

st.markdown(f"<h2 style='font-family: {FONT_FAMILY}; color: {COLORS['text']};'>📊 Economic Impact Dashboard</h2>", 
    unsafe_allow_html=True)
st.markdown(
    f"<span style='color: {COLORS['muted']}; font-size: {FONT_SIZES['body']}px; "
    f"font-family: {FONT_FAMILY};'>"
    f"Community Area 32 (Loop) — {filtered_df['year_month'].min()} → {filtered_df['year_month'].max()} — "
    f"{len(filtered_df):,} block-month records across {filtered_df['block_id'].nunique()} blocks</span>",
    unsafe_allow_html=True
)

st.divider()

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "🔫 Crime Trends", "🏥 Business Health", "🔗 Correlations", "📍 Top Blocks"])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================

with tab1:
    # Metrics row (using filtered data)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"<h5 style='font-family: {FONT_FAMILY}; margin:0; color: {COLORS['muted']};'>Total Crimes (All Time)</h5>", unsafe_allow_html=True)
        st.metric("", f"{int(filtered_df['total_crimes'].sum()):,}")
        if len(filtered_df) < len(df):
            st.caption(f"({len(df) - len(filtered_df):,} records filtered out)")
    
    with c2:
        st.markdown(f"<h5 style='font-family: {FONT_FAMILY}; margin:0; color: {COLORS['muted']};'>Unique Blocks</h5>", unsafe_allow_html=True)
        st.metric("", filtered_df["block_id"].nunique())
    
    with c3:
        st.markdown(f"<h5 style='font-family: {FONT_FAMILY}; margin:0; color: {COLORS['muted']};'>Average BHS</h5>", unsafe_allow_html=True)
        st.metric("", f"{filtered_df['business_health_score'].mean():.1f}")
    
    with c4:
        st.markdown(f"<h5 style='font-family: {FONT_FAMILY}; margin:0; color: {COLORS['muted']};'>Months Analyzed</h5>", unsafe_allow_html=True)
        st.metric("", filtered_df["year_month"].nunique())
    
    with c5:
        st.markdown(f"<h5 style='font-family: {FONT_FAMILY}; margin:0; color: {COLORS['muted']};'>Records After Filter</h5>", unsafe_allow_html=True)
        st.metric("", f"{len(filtered_df):,}")

    st.markdown("")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=filtered_monthly["year_month_dt"], y=filtered_monthly["total_crimes"],
                   name="Total Crimes", marker_color=COLORS["blue"], opacity=0.6),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=filtered_monthly["year_month_dt"], y=filtered_monthly["avg_bhs"],
                       name="Avg BHS", line=dict(color=COLORS["green"], width=2.5)),
            secondary_y=True,
        )
        fig.update_layout(
            **PLOT_LAYOUT,
            title={
                'text': "Monthly Crime Volume & Business Health Score",
                'font': {
                    'family': FONT_FAMILY, 
                    'size': FONT_SIZES["heading"],
                    'color': COLORS["text"]  # Added title color
                }
            },
            yaxis_title="Crimes", 
            yaxis2_title="BHS (0-100)", 
            height=400,
            legend=dict(
                bgcolor=COLORS["surface"],
                bordercolor=COLORS["border"],
                borderwidth=1,
                font=dict(family=FONT_FAMILY, size=FONT_SIZES["small"], color=COLORS["text"]),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        fig.update_yaxes(range=[25, 45], secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📊 Left axis (blue bars): Total monthly crimes | Right axis (green line): Average Business Health Score")

    with col2:
        crime_types = {
            "Theft": int(filtered_df["theft_count"].sum()),
            "Violent": int(filtered_df["violent_crime_count"].sum()),
            "Property": int(filtered_df["property_crime_count"].sum()),
            "Fraud": int(filtered_df["fraud_count"].sum()),
            "Narcotics": int(filtered_df["narcotics_count"].sum()),
            "Weapons": int(filtered_df["weapons_count"].sum()),
        }
        fig_pie = px.pie(
            names=list(crime_types.keys()),
            values=list(crime_types.values()),
            color_discrete_sequence=[COLORS["blue"], COLORS["red"], COLORS["orange"],
                                     COLORS["purple"], COLORS["cyan"], COLORS["pink"]],
            hole=0.45,
        )
        fig_pie.update_layout(
            **PLOT_LAYOUT, 
            title={
                'text': "Crime Type Distribution",
                'font': {
                    'family': FONT_FAMILY, 
                    'size': FONT_SIZES["heading"],
                    'color': COLORS["text"]  # Added title color
                },
               # 'y': 0.98,  # Move title down slightly
               # 'x': 0.5,
               # 'xanchor': 'right',
               # 'yanchor': 'top'
            },
            height=400,
            showlegend=True,
            legend=dict(
                bgcolor=COLORS["surface"],
                bordercolor=COLORS["border"],
                borderwidth=1,
                font=dict(family=FONT_FAMILY, size=FONT_SIZES["small"], color=COLORS["text"]),
                orientation="h",
                yanchor="bottom",
                y=1.15,
                xanchor="center",
                x=0.5
            )#,
            #margin=dict(t=80)
        )

        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption("📊 Distribution of crime types across filtered data")

    col3, col4 = st.columns(2)

    with col3:
        cta_data = filtered_monthly[filtered_monthly["cta_ridership"] > 0]
        fig_cta = go.Figure()
        fig_cta.add_trace(go.Scatter(
            x=cta_data["year_month_dt"], y=cta_data["cta_ridership"],
            fill="tozeroy", fillcolor=f"rgba(6, 182, 212, 0.15)",
            line=dict(color=COLORS["cyan"], width=2), name="CTA Ridership"
        ))
        fig_cta.update_layout(
            **PLOT_LAYOUT, 
            title={
                'text': "CTA Loop Ridership (Monthly)",
                'font': {
                    'family': FONT_FAMILY, 
                    'size': FONT_SIZES["heading"],
                    'color': COLORS["text"]  # Added title color
                }
            },
            yaxis_title="Riders", 
            height=350,
            legend=dict(
                bgcolor=COLORS["surface"],
                bordercolor=COLORS["border"],
                borderwidth=1,
                font=dict(family=FONT_FAMILY, size=FONT_SIZES["small"], color=COLORS["text"]),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        st.plotly_chart(fig_cta, use_container_width=True)
        st.caption("📊 Monthly CTA ridership volume in the Loop area")

    with col4:
        fig_311 = go.Figure()
        fig_311.add_trace(go.Bar(x=filtered_monthly["year_month_dt"], y=filtered_monthly["total_311"],
                                 name="311 Requests", marker_color=COLORS["orange"], opacity=0.7))
        fig_311.add_trace(go.Bar(x=filtered_monthly["year_month_dt"], y=filtered_monthly["graffiti"],
                                 name="Graffiti", marker_color=COLORS["pink"], opacity=0.7))
        fig_311.update_layout(
            **PLOT_LAYOUT, 
            title={
                'text': "311 Requests & Graffiti Reports",
                'font': {
                    'family': FONT_FAMILY, 
                    'size': FONT_SIZES["heading"],
                    'color': COLORS["text"]  # Added title color
                }
            },
            barmode="group", 
            yaxis_title="Count", 
            height=350,
            legend=dict(
                bgcolor=COLORS["surface"],
                bordercolor=COLORS["border"],
                borderwidth=1,
                font=dict(family=FONT_FAMILY, size=FONT_SIZES["small"], color=COLORS["text"]),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        st.plotly_chart(fig_311, use_container_width=True)
        st.caption("📊 Monthly 311 service requests and graffiti reports")

# ============================================================================
# TAB 2: CRIME TRENDS
# ============================================================================

with tab2:
    fig_trends = go.Figure()
    fig_trends.add_trace(go.Scatter(x=filtered_monthly["year_month_dt"], y=filtered_monthly["theft_crimes"],
                                    name="Theft", line=dict(color=COLORS["blue"], width=2)))
    fig_trends.add_trace(go.Scatter(x=filtered_monthly["year_month_dt"], y=filtered_monthly["violent_crimes"],
                                    name="Violent", line=dict(color=COLORS["red"], width=2)))
    fig_trends.add_trace(go.Scatter(x=filtered_monthly["year_month_dt"], y=filtered_monthly["property_crimes"],
                                    name="Property", line=dict(color=COLORS["orange"], width=2)))
    fig_trends.update_layout(
        **PLOT_LAYOUT, 
        title={
            'text': "Crime Type Trends Over Time",
            'font': {
                'family': FONT_FAMILY, 
                'size': FONT_SIZES["heading"],
                'color': COLORS["text"]  # Added title color
            }
        },
        yaxis_title="Monthly Count", 
        height=400,
        legend=dict(
            bgcolor=COLORS["surface"],
            bordercolor=COLORS["border"],
            borderwidth=1,
            font=dict(family=FONT_FAMILY, size=FONT_SIZES["small"], color=COLORS["text"]),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    st.plotly_chart(fig_trends, use_container_width=True)
    st.caption("📈 Trends for major crime categories over the analysis period")

    col1, col2 = st.columns(2)

    with col1:
        fig_cb = make_subplots(specs=[[{"secondary_y": True}]])
        fig_cb.add_trace(
            go.Scatter(x=filtered_monthly["year_month_dt"], y=filtered_monthly["total_crimes"],
                       name="Crimes", line=dict(color=COLORS["red"], width=2)),
            secondary_y=False,
        )
        fig_cb.add_trace(
            go.Scatter(x=filtered_monthly["year_month_dt"], y=filtered_monthly["active_businesses"],
                       name="Active Businesses", line=dict(color=COLORS["green"], width=2)),
            secondary_y=True,
        )
        fig_cb.update_layout(
            **PLOT_LAYOUT, 
            title={
                'text': "Crime vs Active Businesses",
                'font': {
                    'family': FONT_FAMILY, 
                    'size': FONT_SIZES["heading"],
                    'color': COLORS["text"]  # Added title color
                }
            },
            height=350,
            legend=dict(
                bgcolor=COLORS["surface"],
                bordercolor=COLORS["border"],
                borderwidth=1,
                font=dict(family=FONT_FAMILY, size=FONT_SIZES["small"], color=COLORS["text"]),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )

        st.plotly_chart(fig_cb, use_container_width=True)
        st.caption("📊 Relationship between crime volume and active business count")

    with col2:
        fig_sl = make_subplots(specs=[[{"secondary_y": True}]])
        fig_sl.add_trace(
            go.Bar(x=filtered_monthly["year_month_dt"], y=filtered_monthly["streetlight_issues"],
                   name="Streetlight Issues", marker_color=COLORS["orange"], opacity=0.5),
            secondary_y=True,
        )
        fig_sl.add_trace(
            go.Scatter(x=filtered_monthly["year_month_dt"], y=filtered_monthly["total_crimes"],
                       name="Crimes", line=dict(color=COLORS["red"], width=2)),
            secondary_y=False,
        )
        fig_sl.update_layout(
            **PLOT_LAYOUT, 
            title={
                'text': "Streetlight Issues vs Crime",
                'font': {
                    'family': FONT_FAMILY, 
                    'size': FONT_SIZES["heading"],
                    'color': COLORS["text"]  # Added title color
                }
            },
            height=350,
            legend=dict(
                bgcolor=COLORS["surface"],
                bordercolor=COLORS["border"],
                borderwidth=1,
                font=dict(family=FONT_FAMILY, size=FONT_SIZES["small"], color=COLORS["text"]),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )

        st.plotly_chart(fig_sl, use_container_width=True)
        st.caption("📊 Streetlight maintenance issues compared to crime trends")

    # Year-over-year comparison
    st.markdown(f"<h3 style='font-family: {FONT_FAMILY}; color: {COLORS['text']};'>Year-over-Year Comparison</h3>", 
    unsafe_allow_html=True)
    filtered_monthly["year"] = filtered_monthly["year_month_dt"].dt.year
    filtered_monthly["month"] = filtered_monthly["year_month_dt"].dt.month
    year_select = st.multiselect("Select years to compare", sorted(filtered_monthly["year"].unique()),
                                 default=sorted(filtered_monthly["year"].unique())[-3:] if len(filtered_monthly["year"].unique()) >= 3 else sorted(filtered_monthly["year"].unique()),
                                 key="yoy_years")
    if year_select:
        yoy = filtered_monthly[filtered_monthly["year"].isin(year_select)]
        fig_yoy = px.line(yoy, x="month", y="total_crimes", color="year",
                          color_discrete_sequence=[COLORS["blue"], COLORS["red"], COLORS["green"],
                                                   COLORS["purple"], COLORS["orange"], COLORS["cyan"]],
                          labels={"month": "Month", "total_crimes": "Total Crimes", "year": "Year"})
        fig_yoy.update_layout(
            **PLOT_LAYOUT, 
            title={
                'text': "Year-over-Year Crime Comparison",
                'font': {
                    'family': FONT_FAMILY, 
                    'size': FONT_SIZES["heading"],
                    'color': COLORS["text"]  # Added title color
                }
            },
            height=400,
            legend=dict(
                bgcolor=COLORS["surface"],
                bordercolor=COLORS["border"],
                borderwidth=1,
                font=dict(family=FONT_FAMILY, size=FONT_SIZES["small"], color=COLORS["text"]),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        st.plotly_chart(fig_yoy, use_container_width=True)

# ============================================================================
# TAB 3: BUSINESS HEALTH
# ============================================================================

with tab3:
    fig_bhs = go.Figure()
    fig_bhs.add_trace(go.Scatter(
        x=filtered_monthly["year_month_dt"], y=filtered_monthly["avg_bhs"],
        fill="tozeroy", fillcolor=f"rgba(16, 185, 129, 0.15)",
        line=dict(color=COLORS["green"], width=2.5), name="Avg BHS"
    ))
    fig_bhs.update_layout(
        **PLOT_LAYOUT, 
        title={
            'text': "Business Health Score Over Time",
            'font': {
                'family': FONT_FAMILY, 
                'size': FONT_SIZES["heading"],
                'color': COLORS["text"]  # Added title color
            }
        },
        yaxis_title="BHS (0–100)", 
        yaxis_range=[25, 45], 
        height=400,
        legend=dict(
            bgcolor=COLORS["surface"],
            bordercolor=COLORS["border"],
            borderwidth=1,
            font=dict(family=FONT_FAMILY, size=FONT_SIZES["small"], color=COLORS["text"]),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    st.plotly_chart(fig_bhs, use_container_width=True)
    st.caption("📈 Average Business Health Score trend (higher = better)")

    col1, col2 = st.columns(2)

    with col1:
        cta_bhs = filtered_monthly[filtered_monthly["cta_ridership"] > 0]
        fig_scatter = px.scatter(
            cta_bhs, x="cta_ridership", y="avg_bhs",
            hover_data=["year_month"],
            labels={"cta_ridership": "CTA Monthly Ridership", "avg_bhs": "Avg BHS"},
            color_discrete_sequence=[COLORS["cyan"]],
        )
        fig_scatter.update_layout(
            **PLOT_LAYOUT, 
            title={
                'text': "BHS vs CTA Ridership",
                'font': {'family': FONT_FAMILY, 'size': FONT_SIZES["heading"], 'color': COLORS["text"]}
            },
            height=400,
            legend=dict(
                bgcolor=COLORS["surface"],
                bordercolor=COLORS["border"],
                borderwidth=1,
                font=dict(family=FONT_FAMILY, size=FONT_SIZES["small"], color=COLORS["text"]),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("📊 Correlation between transit ridership and business health")

    with col2:
        covid = filtered_monthly[(filtered_monthly["year_month_dt"] >= "2019-01-01") & (filtered_monthly["year_month_dt"] <= "2021-06-01")]
        if len(covid) > 0:
            fig_covid = make_subplots(specs=[[{"secondary_y": True}]])
            fig_covid.add_trace(
                go.Bar(x=covid["year_month_dt"], y=covid["total_crimes"],
                       name="Crimes", marker_color=COLORS["red"], opacity=0.5),
                secondary_y=False,
            )
            fig_covid.add_trace(
                go.Scatter(x=covid["year_month_dt"], y=covid["avg_bhs"],
                           name="BHS", line=dict(color=COLORS["green"], width=2.5)),
                secondary_y=True,
            )
            fig_covid.update_layout(
                **PLOT_LAYOUT, 
                title={
                    'text': "COVID Impact: Crime & BHS (2019–2021)",
                    'font': {'family': FONT_FAMILY, 'size': FONT_SIZES["heading"], 'color': COLORS["text"]}
                },
                height=400,
                legend=dict(
                    bgcolor=COLORS["surface"],
                    bordercolor=COLORS["border"],
                    borderwidth=1,
                    font=dict(family=FONT_FAMILY, size=FONT_SIZES["small"], color=COLORS["text"]),
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            )
            fig_covid.update_yaxes(range=[28, 42], secondary_y=True)
            st.plotly_chart(fig_covid, use_container_width=True)
            st.caption("📊 COVID-19 pandemic period analysis: crime vs business health")
        else:
            st.info("No data available for COVID period with current filters")

    fig_hist = px.histogram(filtered_df, x="business_health_score", nbins=40,
                            color_discrete_sequence=[COLORS["green"]],
                            labels={"business_health_score": "Business Health Score"})
    fig_hist.update_layout(
        **PLOT_LAYOUT, 
        title={
            'text': "Distribution of Business Health Score Across All Block-Months",
            'font': {'family': FONT_FAMILY, 'size': FONT_SIZES["heading"], 'color': COLORS["text"]}
        },
        yaxis_title="Count", 
        height=350,
        legend=dict(
            bgcolor=COLORS["surface"],
            bordercolor=COLORS["border"],
            borderwidth=1,
            font=dict(family=FONT_FAMILY, size=FONT_SIZES["small"], color=COLORS["text"]),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption("📊 Frequency distribution of BHS values across all blocks and months")

# ============================================================================
# TAB 4: CORRELATIONS
# ============================================================================

with tab4:
    if len(filtered_correlations) > 0:
        corr_df = filtered_correlations.reset_index()
        corr_df.columns = ["Feature", "Correlation"]
        corr_df["Color"] = corr_df["Correlation"].apply(lambda x: COLORS["green"] if x >= 0 else COLORS["red"])
        corr_df["Abs"] = corr_df["Correlation"].abs()
        corr_df = corr_df.sort_values("Correlation")

        fig_corr = go.Figure()
        fig_corr.add_trace(go.Bar(
            y=corr_df["Feature"], x=corr_df["Correlation"],
            orientation="h",
            marker_color=corr_df["Color"],
            text=corr_df["Correlation"].apply(lambda x: f"{x:+.3f}"),
            textposition="outside",
            textfont=dict(family=FONT_FAMILY, size=FONT_SIZES["tiny"]),
        ))
        fig_corr.update_layout(
            **PLOT_LAYOUT,
            title={
                'text': "Feature Correlations with Business Health Score",
                'font': {'family': FONT_FAMILY, 'size': FONT_SIZES["heading"],'color': COLORS["text"]}
            },
            xaxis_title="Pearson Correlation",
            height=550,
            legend=dict(
                bgcolor=COLORS["surface"],
                bordercolor=COLORS["border"],
                borderwidth=1,
                font=dict(family=FONT_FAMILY, size=FONT_SIZES["small"], color=COLORS["text"]),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        fig_corr.update_xaxes(range=[-0.35, 1.0])
        fig_corr.update_yaxes(tickfont=dict(family=FONT_FAMILY, size=FONT_SIZES["small"]))

        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("📊 Positive values (green) indicate features that boost BHS, negative (red) indicate features that reduce BHS")

        st.markdown(f"""
        <div style="background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 24px; margin-top: 8px; font-family: {FONT_FAMILY};">
            <h4 style="color: #e2e8f0; margin-bottom: 12px; font-family: {FONT_FAMILY};">🔑 Key Insights</h4>
            <ul style="color: #94a3b8; line-height: 2; font-family: {FONT_FAMILY};">
                <li><strong style="color: {COLORS['green']};">Business diversity</strong> — Strongest predictor. Blocks with varied business types have higher BHS.</li>
                <li><strong style="color: {COLORS['green']};">Active business count</strong> — More businesses = higher BHS, as expected.</li>
                <li><strong style="color: {COLORS['green']};">CTA ridership</strong> — Foot traffic is a strong indicator of vitality.</li>
                <li><strong style="color: {COLORS['red']};">Crimes per business</strong> — The strongest negative signal. Crime density relative to businesses hurts block health.</li>
                <li><strong style="color: {COLORS['red']};">Night crime ratio</strong> — Blocks with more nighttime crime have a higher public safety penalty.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<h4 style='font-family: {FONT_FAMILY}; color: {COLORS['text']};'>Full Feature Correlation Heatmap</h4>", 
    unsafe_allow_html=True)
        st.caption("📊 Matrix showing correlations between all pairs of features (darker green = stronger positive, darker red = stronger negative)")
        
        heatmap_features = [
            "total_crimes", "violent_crime_count", "theft_count", "property_crime_count",
            "arrest_rate", "night_crime_ratio", "active_business_count",
            "business_diversity_index", "crimes_per_business", "cta_total_loop_ridership",
            "total_311_requests", "streetlight_issues", "graffiti_reports",
            "business_health_score"
        ]
        existing_hm = [f for f in heatmap_features if f in filtered_df.columns]
        if len(existing_hm) > 1:
            corr_matrix = filtered_df[existing_hm].corr()

            fig_hm = px.imshow(
                corr_matrix.round(2),
                text_auto=True,
                color_continuous_scale=["#ef4444", "#1e293b", "#10b981"],
                zmin=-1, zmax=1,
            )
            fig_hm.update_layout(
                **PLOT_LAYOUT, 
                title={
                    'text': "Correlation Heatmap",
                    'font': {'family': FONT_FAMILY, 'size': FONT_SIZES["heading"], 'color': COLORS["text"]}
                },
                height=600
            )
            fig_hm.update_xaxes(tickangle=45, tickfont_size=10)
            fig_hm.update_yaxes(tickfont_size=10)
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.info("Insufficient data for correlation heatmap with current filters")
    else:
        st.info("No correlation data available with current filters")

# ============================================================================
# TAB 5: TOP BLOCKS
# ============================================================================

with tab5:
    if len(filtered_top_blocks) > 0:
        top10 = filtered_top_blocks.head(10)

        # Create a more interpretable chart
        fig_blocks = make_subplots(
            specs=[[{"secondary_y": True}]],
        )

        # Add Total Crimes bars
        fig_blocks.add_trace(
            go.Bar(
                x=top10["block_id"], 
                y=top10["total_crimes"],
                name="Total Crimes", 
                marker_color=COLORS["red"], 
                opacity=1,
                hovertemplate="<b>Block: %{x}</b><br>" +
                            "Total Crimes: %{y:,}<br>" +
                            "<extra></extra>"
            ),
            secondary_y=False,
        )

        # Add BHS bars
        fig_blocks.add_trace(
            go.Bar(
                x=top10["block_id"], 
                y=top10["avg_bhs"],
                name="Block Health Score", 
                marker_color=COLORS["green"], 
                opacity=1,
                hovertemplate="<b>Block: %{x}</b><br>" +
                            "Avg BHS: %{y:.1f}<br>" +
                            "<extra></extra>"
            ),
            secondary_y=True,
        )

        # Add annotations to highlight the relationship
        for i, row in top10.iterrows():
            fig_blocks.add_annotation(
                x=row["block_id"],
                y=row["total_crimes"],
                yref="y",
                text=f"BHS: {row['avg_bhs']:.1f}",
                showarrow=False,
                yshift=10,
                font=dict(family=FONT_FAMILY, size=FONT_SIZES["tiny"], color="#FFFFFF"),
                bgcolor="#000000",
                bordercolor="rgba(0,0,0,0)",
                opacity=0.7
            )

        # Update layout with clearer labels and explanation
        fig_blocks.update_layout(
            **PLOT_LAYOUT,
            title={
                'text': "Top 10 Crime Blocks: Crime Volume vs Block Health Score",
                'font': {'family': FONT_FAMILY, 'size': FONT_SIZES["heading"], 'color': COLORS["text"]}
            },
            barmode="group",
            height=450,
            hovermode="x unified",
            legend=dict(
                bgcolor=COLORS["surface"],
                bordercolor=COLORS["border"],
                borderwidth=1,
                font=dict(family=FONT_FAMILY, size=FONT_SIZES["small"], color=COLORS["text"]),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )

        # Add explanatory subtitle
        fig_blocks.add_annotation(
            text="⬆️ Higher bars = more crime | 🟢 Higher BHS = better block health",
            xref="paper", yref="paper",
            x=0.5, y=1.08,
            showarrow=False,
            font=dict(family=FONT_FAMILY, size=FONT_SIZES["small"], color=COLORS["muted"])
        )

        # Update axes labels
        fig_blocks.update_xaxes(
            tickangle=-30, 
            tickfont_size=FONT_SIZES["tiny"],
            title_text="Block ID",
            title_font=dict(family=FONT_FAMILY, size=FONT_SIZES["small"])
        )

        fig_blocks.update_yaxes(
            title_text="Total Number of Crimes",
            range=[0, max(top10["total_crimes"]) * 1.1] if len(top10) > 0 else [0, 1],
            secondary_y=False,
            gridcolor=COLORS["muted"],
            griddash="dot",
            title_font=dict(family=FONT_FAMILY, size=FONT_SIZES["small"])
        )

        fig_blocks.update_yaxes(
            title_text="Average Block Health Score (BHS)",
            range=[0, 55],
            secondary_y=True,
            gridcolor=COLORS["muted"],
            griddash="dot",
            title_font=dict(family=FONT_FAMILY, size=FONT_SIZES["small"])
        )

        # Add a reference line for BHS threshold
        fig_blocks.add_hline(
            y=40, 
            line_dash="dash", 
            line_color=COLORS["green"],
            opacity=0.3,
            line_width=1.5,
            secondary_y=True,
            annotation_text="Target BHS",
            annotation_position="top right",
            annotation_font=dict(family=FONT_FAMILY, size=FONT_SIZES["tiny"])
        )

        st.plotly_chart(fig_blocks, use_container_width=True)
        st.caption("📊 Top 10 blocks by crime volume: comparing crime counts (red) with Business Health Scores (green)")

        # Add interpretation text below the chart
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div style='background-color: {COLORS["surface"]}; padding: 10px; border-radius: 5px; border-left: 4px solid {COLORS["red"]}; font-family: {FONT_FAMILY};'>
                <h5 style='margin:0; color: {COLORS["red"]}; font-family: {FONT_FAMILY};'>🔴 High Crime Blocks</h5>
                <small style='color: {COLORS["muted"]}; font-family: {FONT_FAMILY};'>These blocks have the highest crime volumes in the dataset</small>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style='background-color: {COLORS["surface"]}; padding: 10px; border-radius: 5px; border-left: 4px solid {COLORS["green"]}; font-family: {FONT_FAMILY};'>
                <h5 style='margin:0; color: {COLORS["green"]}; font-family: {FONT_FAMILY};'>🟢 Block Health Score</h5>
                <small style='color: {COLORS["muted"]}; font-family: {FONT_FAMILY};'>Higher BHS indicates better overall block health</small>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style='background-color: {COLORS["surface"]}; padding: 10px; border-radius: 5px; border-left: 4px solid {COLORS["blue"]}; font-family: {FONT_FAMILY};'>
                <h5 style='margin:0; color: {COLORS["blue"]}; font-family: {FONT_FAMILY};'>📊 Interpretation</h5>
                <small style='color: {COLORS["muted"]}; font-family: {FONT_FAMILY};'>Blocks with high crime often show lower BHS</small>
            </div>
            """, unsafe_allow_html=True)

        # Scatter plot
        if len(filtered_top_blocks) > 5:
            filtered_top_blocks["violence_level"] = pd.cut(
                filtered_top_blocks["avg_violent_ratio"],
                bins=[0, 0.2, 0.3, 1.0],
                labels=["Low (<20%)", "Medium (20-30%)", "High (>30%)"]
            )
            fig_scatter2 = px.scatter(
                filtered_top_blocks, x="avg_businesses", y="total_crimes",
                color="violence_level",
                color_discrete_map={"Low (<20%)": COLORS["green"], "Medium (20-30%)": COLORS["orange"], "High (>30%)": COLORS["red"]},
                size="avg_bhs", size_max=25,
                hover_data=["block_id", "avg_bhs", "avg_violent_ratio"],
                labels={"avg_businesses": "Avg Active Businesses", "total_crimes": "Total Crimes",
                        "violence_level": "Violent Crime Level"},
            )
            fig_scatter2.update_layout(
                **PLOT_LAYOUT, 
                title={
                    'text': "Block Crime vs Business Density (sized by BHS)",
                    'font': {'family': FONT_FAMILY, 'size': FONT_SIZES["heading"], 'color': COLORS["text"]}
                },
                height=450,
                legend=dict(
                    bgcolor=COLORS["surface"],
                    bordercolor=COLORS["border"],
                    borderwidth=1,
                    font=dict(family=FONT_FAMILY, size=FONT_SIZES["small"], color=COLORS["text"]),
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            )
            st.plotly_chart(fig_scatter2, use_container_width=True)
            st.caption("📊 Each point represents a block: X-axis = businesses, Y-axis = crimes, Color = violent crime level, Size = BHS")

        # Map
        st.markdown(f"<h3 style='font-family: {FONT_FAMILY}; color: {COLORS['text']};'>🗺️ Crime Burden Map</h3>", 
    unsafe_allow_html=True)
        valid_blocks = filtered_top_blocks.dropna(subset=["lat", "lon"])
        if len(valid_blocks) > 0:
            fig_map = px.scatter_mapbox(
                valid_blocks, lat="lat", lon="lon",
                size="avg_crimes_per_biz", size_max=30,
                color="avg_bhs",
                color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
                hover_name="block_id",
                hover_data={
                    "avg_crimes_per_biz": ":.2f",
                    "total_crimes": True, 
                    "avg_bhs": ":.1f", 
                    "avg_businesses": ":.1f", 
                    "lat": False, 
                    "lon": False
                },
                mapbox_style="carto-darkmatter",
                zoom=14,
                center={"lat": valid_blocks["lat"].mean(), "lon": valid_blocks["lon"].mean()},
            )
            fig_map.update_layout(
                margin=dict(l=0, r=0, t=40, b=0),
                height=500,
                title={
                    'text': "Top 30 Crime Blocks (color = BHS, size = Crimes per Business)",
                    'font': {'family': FONT_FAMILY, 'size': FONT_SIZES["heading"], 'color': COLORS["text"]}
                },
                paper_bgcolor="#111827",
                font=dict(family=FONT_FAMILY, color="#e2e8f0"),
            )
            st.plotly_chart(fig_map, use_container_width=True)
            st.caption("🗺️ Geographic distribution: Green = healthier blocks, Red = struggling blocks. Larger circles = higher crime density per business")

        # Data table
        st.markdown(f"<h3 style='font-family: {FONT_FAMILY}; color: {COLORS['text']};'>📋 Block Details</h3>", 
    unsafe_allow_html=True)
        st.caption("Detailed metrics for top 30 blocks (sorted by crime volume)")
        display_cols = ["block_id", "total_crimes", "avg_crimes_per_biz", "avg_bhs", "avg_businesses", "avg_violent_ratio", "avg_night_ratio"]
        if all(col in filtered_top_blocks.columns for col in display_cols):
            st.dataframe(
                filtered_top_blocks[display_cols].rename(columns={
                    "block_id": "Block", 
                    "total_crimes": "Total Crimes", 
                    "avg_crimes_per_biz": "Crimes/Biz",
                    "avg_bhs": "Avg BHS",
                    "avg_businesses": "Avg Businesses", 
                    "avg_violent_ratio": "Violent %", 
                    "avg_night_ratio": "Night %"
                }).style.format({
                    "Crimes/Biz": "{:.2f}",
                    "Avg BHS": "{:.1f}", 
                    "Avg Businesses": "{:.1f}",
                    "Violent %": "{:.1%}", 
                    "Night %": "{:.1%}"
                }).background_gradient(subset=["Crimes/Biz"], cmap="Reds")
                .background_gradient(subset=["Avg BHS"], cmap="Greens"),
                use_container_width=True,
                height=400,
            )
    else:
        st.info("No block data available with current filters")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption("🔄 Hover over any chart for detailed values | Use filters in sidebar to explore specific segments")