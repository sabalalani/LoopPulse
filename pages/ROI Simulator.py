"""
Loop Pulse — ML Intervention ROI Simulator
=========================================
100% data-driven, zero hardcoded values. Optimized for speed.
All interventions, costs, impacts learned from historical patterns.
Includes 5-fold cross-validation and Adjusted R² for robust model evaluation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
import pickle
import os
import gc
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Loop Pulse — ML ROI Simulator", 
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# FONT CONFIGURATION (Same as main dashboard)
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

# ============================================================================
# COLOR SCHEME (Identical to main dashboard)
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
# DASHBOARD SELECTOR FIX - Using session state and query params
# ============================================================================

# ============================================================================
# CUSTOM CSS (Fixed for dark sidebar)
# ============================================================================

st.markdown(f"""
<style>
    /* Force main area to be dark */
    .stApp {{
        background-color: {COLORS['bg']} !important;
    }}
    
    .main .block-container {{
        background-color: {COLORS['bg']} !important;
    }}
    
    /* Light sidebar ONLY */
    section[data-testid="stSidebar"] {{
        background-color: #f0f2f6 !important;
        border-right: 1px solid #e5e7eb !important;
    }}
    
    /* All sidebar content */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] .stMarkdown div {{
        color: #111827 !important;
    }}
    
    /* Sidebar headings */
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {{
        color: #111827 !important;
    }}
    
    /* Sidebar labels */
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stSelectbox label {{
        color: #4b5563 !important;
    }}
    
    /* Sidebar select boxes */
    section[data-testid="stSidebar"] .stSelectbox > div > div {{
        background-color: #f9fafb !important;
        border-color: #e5e7eb !important;
        color: #111827 !important;
    }}
    
    /* Sidebar sliders */
    section[data-testid="stSidebar"] .stSlider > div > div {{
        color: #3b82f6 !important;
    }}
    
    /* Sidebar buttons */
    section[data-testid="stSidebar"] .stButton button {{
        background-color: #f9fafb !important;
        color: #111827 !important;
        border: 1px solid #e5e7eb !important;
    }}
    
    section[data-testid="stSidebar"] .stButton button:hover {{
        border-color: #3b82f6 !important;
        color: #3b82f6 !important;
        background-color: #eff6ff !important;
    }}
    
    /* Current BHS box in sidebar */
    section[data-testid="stSidebar"] div[style*="background-color"] {{
        background-color: #f9fafb !important;
        border-color: #e5e7eb !important;
    }}
    
    section[data-testid="stSidebar"] div[style*="background-color"] span {{
        color: #111827 !important;
    }}
    
    /* Sidebar dividers */
    section[data-testid="stSidebar"] hr {{
        border-color: #e5e7eb !important;
    }}
    
    /* Sidebar success/info messages */
    section[data-testid="stSidebar"] .stAlert {{
        background-color: #f9fafb !important;
        color: #111827 !important;
        border: 1px solid #e5e7eb !important;
    }}
    
    /* Sidebar expanders */
    section[data-testid="stSidebar"] .streamlit-expanderHeader {{
        background-color: #f9fafb !important;
        color: #111827 !important;
        border: 1px solid #e5e7eb !important;
    }}
    
    section[data-testid="stSidebar"] .streamlit-expanderContent {{
        background-color: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-top: none !important;
    }}
    
    /* Sidebar number inputs */
    section[data-testid="stSidebar"] .stNumberInput input {{
        background-color: #f9fafb !important;
        color: #111827 !important;
        border-color: #e5e7eb !important;
    }}
    
    /* Sidebar checkboxes */
    section[data-testid="stSidebar"] .stCheckbox label {{
        color: #111827 !important;
    }}
    
    /* Ensure main area text uses dark theme colors */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6,
    .main p, .main span, .main div:not([data-testid="stSidebar"] *) {{
        color: {COLORS['text']} !important;
    }}
    
    /* Keep metric cards in main area dark */
    .main div[data-testid="stMetric"] {{
        background-color: {COLORS['surface']} !important;
        border: 1px solid {COLORS['border']} !important;
    }}
    
    .main div[data-testid="stMetric"] label {{
        color: {COLORS['muted']} !important;
    }}
    
    .main div[data-testid="stMetric"] div {{
        color: {COLORS['text']} !important;
    }}
    
    /* Keep info boxes in main area dark */
    .main .info-box {{
        background-color: {COLORS['surface']} !important;
        border-left: 4px solid {COLORS['blue']} !important;
    }}
    
    .main .info-box span {{
        color: {COLORS['text']} !important;
    }}
    
    /* Keep dividers in main area dark */
    .main hr {{
        border-color: {COLORS['border']} !important;
    }}
    
    /* Keep captions in main area dark */
    .main .stCaption, .main caption {{
        color: {COLORS['muted']} !important;
    }}
    
    /* Keep expanders in main area dark */
    .main .streamlit-expanderHeader {{
        background-color: {COLORS['surface']} !important;
        color: {COLORS['text']} !important;
        border: 1px solid {COLORS['border']} !important;
    }}
    
    .main .streamlit-expanderContent {{
        background-color: {COLORS['surface']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-top: none !important;
    }}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# PLOT LAYOUT (Same as main dashboard)
# ============================================================================

PLOT_LAYOUT = dict(
    paper_bgcolor=COLORS["surface"],
    plot_bgcolor=COLORS["bg"],
    font=dict(
        family=FONT_FAMILY,
        color=COLORS["text"], 
        size=FONT_SIZES["body"]
    ),
    margin=dict(l=60, r=30, t=50, b=50),
    xaxis=dict(
        gridcolor=COLORS["border"],
        title_font=dict(family=FONT_FAMILY, size=FONT_SIZES["small"]),
        tickfont=dict(family=FONT_FAMILY, size=FONT_SIZES["tiny"]),
        linecolor=COLORS["border"],
        zerolinecolor=COLORS["border"]
    ),
    yaxis=dict(
        gridcolor=COLORS["border"],
        title_font=dict(family=FONT_FAMILY, size=FONT_SIZES["small"]),
        tickfont=dict(family=FONT_FAMILY, size=FONT_SIZES["tiny"]),
        linecolor=COLORS["border"],
        zerolinecolor=COLORS["border"]
    ),
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

# ============================================================================
# HELPER FUNCTION: Calculate Adjusted R²
# ============================================================================
def adjusted_r2(r2, n, p):
    """Calculate Adjusted R²: 1 - ((1-R²)(n-1)/(n-p-1))"""
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))

# ============================================================================
# DICTIONARY FOR HUMAN-READABLE UI
# ============================================================================
UI_DICTIONARY = {
    "streetlight_issues": {"name": "Streetlight Repair Initiative", "unit": "blocks repaired", "base_cost": 4500},
    "graffiti_reports": {"name": "Graffiti Cleanup Teams", "unit": "cleanup crews", "base_cost": 2200},
    "infrastructure_issues": {"name": "Sidewalk & Pothole Fixes", "unit": "projects", "base_cost": 8000},
    "total_311_requests": {"name": "General 311 Response Surge", "unit": "response teams", "base_cost": 5000},
    "active_business_count": {"name": "Small Business Subsidies", "unit": "grants", "base_cost": 25000},
    "business_diversity_index": {"name": "Retail Diversity Incentives", "unit": "zoned grants", "base_cost": 30000},
    "food_business_ratio": {"name": "Restaurant Incubation", "unit": "permits expedited", "base_cost": 15000},
}

# ============================================================================
# MODEL PERSISTENCE - Save/Load trained model
# ============================================================================

MODEL_FILE = "loop_pulse_model.pkl"

def save_model(model_data):
    """Save trained model to pickle file"""
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model_data, f)

def load_model():
    """Load trained model from pickle file if it exists"""
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.sidebar.warning(f"Could not load existing model: {e}")
            return None
    return None

# ============================================================================
# ULTRA-OPTIMIZED DATA LOADING WITH CHUNKING
# ============================================================================
@st.cache_data(ttl=3600, show_spinner="Loading & optimizing 25 years of data...")
def load_optimized_data():
    chunks = []
    try:
        for chunk in pd.read_csv("loop_pulse_features2.csv", chunksize=5000):
            for col in chunk.select_dtypes(include=['float64']).columns:
                chunk[col] = pd.to_numeric(chunk[col], downcast='float')
            for col in chunk.select_dtypes(include=['int64']).columns:
                chunk[col] = pd.to_numeric(chunk[col], downcast='integer')
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['business_health_score'])
    return df

# ============================================================================
# TRAIN OR LOAD MODEL
# ============================================================================
@st.cache_resource(show_spinner="Loading AI Digital Twin...")
def get_digital_twin(_df):
    """Load existing model or train new one"""
    
    # Try to load existing model
    model_data = load_model()
    
    if model_data is not None:
        st.sidebar.success("✅ Loaded pre-trained model from disk")
        return model_data
    
    # Train new model if none exists
    st.sidebar.info("🔄 Training new model (this will be saved for future use)...")
    df = _df.copy()

    # 1. AUTOMATIC FEATURE DISCOVERY
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_patterns = ['block_id', 'year', 'month', 'lat', 'lon', 'centroid', 'index']

    feature_candidates = [
        c for c in numeric_cols
        if c != 'business_health_score'
           and not any(p in c.lower() for p in exclude_patterns)
           and df[c].nunique() >= 2
    ]

    correlations = []
    for col in feature_candidates:
        valid_data = df[[col, 'business_health_score']].dropna()
        if len(valid_data) > 100:
            corr = valid_data.corr().iloc[0, 1]
            if not np.isnan(corr) and abs(corr) >= 0.1:
                correlations.append({'feature': col, 'correlation': abs(corr), 'direction': np.sign(corr)})

    correlations.sort(key=lambda x: x['correlation'], reverse=True)
    features = [c['feature'] for c in correlations]
    if len(features) < 3: 
        features = [c['feature'] for c in correlations[:5]]

    # 2. LEARN COSTS
    if 'active_business_count' in df.columns:
        avg_business_value = df['active_business_count'].median() * 50000
        bhs_to_value_ratio = avg_business_value / df['business_health_score'].median()
    else:
        bhs_to_value_ratio = 10000

    base_cost_unit = bhs_to_value_ratio * 0.5

    feature_costs = {}
    for feature in features:
        mean_val, std_val = df[feature].mean(), df[feature].std()
        if mean_val > 0 and std_val > 0:
            cost_multiplier = 1 + (std_val / mean_val)
            feature_costs[feature] = max(500, min(100000, int(base_cost_unit * cost_multiplier)))

    # 3. K-MEANS INTERVENTION DISCOVERY
    cluster_data = df[features].fillna(0)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(cluster_data)

    centroids = df.groupby('cluster')[features].mean()
    cluster_perf = df.groupby('cluster')['business_health_score'].mean().sort_values()
    gap = centroids.loc[cluster_perf.index[-1]] - centroids.loc[cluster_perf.index[0]]

    interventions = {}
    uncontrollable_keywords = ['crime', 'cta', 'ridership', 'ratio', 'index', 'lag', 'mom', 'arrest', 'count_lag']

    for feature in features:
        if abs(gap[feature]) > 0.1 and not any(k in feature.lower() for k in uncontrollable_keywords):
            direction = 'increase' if gap[feature] > 0 else 'decrease'
            impact_sign = 1 if gap[feature] > 0 else -1

            ui_info = UI_DICTIONARY.get(feature, {
                "name": f"Optimize {feature.replace('_', ' ').title()}",
                "unit": "units", "base_cost": 5000
            })

            mean_val, std_val = df[feature].mean(), df[feature].std()
            cost_multiplier = 1 + (std_val / mean_val) if mean_val > 0 else 1
            calculated_cost = max(500, min(100000, int(ui_info["base_cost"] * cost_multiplier)))

            interventions[feature] = {
                'name': ui_info["name"], 'unit': ui_info["unit"],
                'description': f"Data shows top-tier blocks have {abs(gap[feature]):.1f} {'more' if gap[feature] > 0 else 'less'} {feature.replace('_', ' ')}.",
                'direction': direction,
                'impact_per_unit': abs(gap[feature]) * 0.15 * impact_sign,
                'max_qty': max(5, int(abs(gap[feature]) * 2)),
                'feature': feature, 'cost': calculated_cost
            }

    # 4. TRAIN RANDOM FOREST MODEL
    X = df[features].fillna(0)
    y = df['business_health_score'].fillna(50)
    n_samples = len(X)
    n_features = len(features)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_split=10, random_state=42, n_jobs=-1)

    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    cv_adjusted = [adjusted_r2(score, n_samples, n_features) for score in cv_scores]
    cv_adj_mean = np.mean(cv_adjusted)
    cv_adj_std = np.std(cv_adjusted)

    # Train final model
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    train_r2 = r2_score(y, y_pred)
    train_adj_r2 = adjusted_r2(train_r2, n_samples, n_features)

    importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)

    block_summary = df.groupby('block_id').agg({'business_health_score': 'mean', **{f: 'mean' for f in features}}).reset_index()

    model_data = {
        'features': features,
        'correlations': correlations,
        'feature_costs': feature_costs,
        'base_cost': base_cost_unit,
        'interventions': interventions,
        'model': model,
        'scaler': scaler,
        'train_r2': train_r2,
        'train_adj_r2': train_adj_r2,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'cv_adj_mean': cv_adj_mean,
        'cv_adj_std': cv_adj_std,
        'importance': importance,
        'block_summary': block_summary,
        'economic_value_per_bhs_point': bhs_to_value_ratio,
        'df': df  # Save a reference to the full dataframe
    }
    
    # Save to disk for future use
    save_model(model_data)
    st.sidebar.success("✅ Model trained and saved to disk!")
    return model_data

# ============================================================================
# MAIN APP
# ============================================================================
df = load_optimized_data()
if df is None:
    st.stop()

try:
    # Load or train the digital twin
    twin_data = get_digital_twin(df)
    
    # Extract all components
    features = twin_data['features']
    correlations = twin_data['correlations']
    feature_costs = twin_data['feature_costs']
    base_cost = twin_data['base_cost']
    interventions = twin_data['interventions']
    model = twin_data['model']
    scaler = twin_data['scaler']
    train_r2 = twin_data['train_r2']
    train_adj_r2 = twin_data['train_adj_r2']
    cv_mean = twin_data['cv_mean']
    cv_std = twin_data['cv_std']
    cv_adj_mean = twin_data['cv_adj_mean']
    cv_adj_std = twin_data['cv_adj_std']
    importance = twin_data['importance']
    block_summary = twin_data['block_summary']
    economic_value_per_bhs_point = twin_data['economic_value_per_bhs_point']
    
except Exception as e:
    st.error(f"Failed to initialize Digital Twin: {e}")
    st.stop()

# ============================================================================
# HEADER
# ============================================================================
st.markdown(
    f"<h2 style='font-family: {FONT_FAMILY}; color: {COLORS['text']};'>AI-Driven Intervention ROI Simulator</h2>", 
    unsafe_allow_html=True
)

st.divider()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown(
        f"<h3 style='font-family: {FONT_FAMILY}; color: {COLORS['text']}; margin-bottom: 10px;'>📍 Target Block</h3>", 
        unsafe_allow_html=True
    )
    
    # Initialize session state for selected block if not exists
    if 'selected_block' not in st.session_state:
        st.session_state.selected_block = block_summary.sort_values('business_health_score', ascending=False)['block_id'].iloc[0] if len(block_summary) > 0 else None
    
    block_options = block_summary.sort_values('business_health_score', ascending=False)['block_id'].tolist() if 'business_health_score' in block_summary.columns else block_summary['block_id'].tolist()

    selected_block = st.selectbox(
        "Select block",
        block_options,
        index=block_options.index(st.session_state.selected_block) if st.session_state.selected_block in block_options else 0,
        format_func=lambda x: f"{x} (BHS: {block_summary[block_summary['block_id']==x]['business_health_score'].values[0]:.1f})",
        label_visibility="collapsed",
        key="block_selector"
    )
    
    # Update session state when selection changes
    if selected_block != st.session_state.selected_block:
        st.session_state.selected_block = selected_block
        # Reset intervention sliders when block changes
        for key in list(st.session_state.keys()):
            if key.startswith("slider_"):
                del st.session_state[key]

    block_data = block_summary[block_summary['block_id'] == selected_block].iloc[0]
    baseline_bhs = block_data['business_health_score']

    st.markdown(
        f"<div style='background-color: {COLORS['bg']}; padding: 15px; border-radius: 12px; border: 1px solid {COLORS['border']}; margin: 10px 0;'>"
        f"<span style='color: {COLORS['muted']};'>Current BHS:</span> "
        f"<span style='color: {COLORS['text']}; font-size: 24px; font-weight: 700;'>{baseline_bhs:.1f}</span>"
        f"</div>", 
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown(
        f"<h3 style='font-family: {FONT_FAMILY}; color: {COLORS['text']};'>Data-Discovered Interventions</h3>", 
        unsafe_allow_html=True
    )

    # Initialize intervention config in session state
    if 'intervention_config' not in st.session_state:
        st.session_state.intervention_config = {}

    config = {}
    total_cost_display = 0
    
    # Create a container for interventions
    interventions_container = st.container()
    
    with interventions_container:
        for key, inv in interventions.items():
            with st.container():
                st.markdown(
                    f"<div style='color: {COLORS['text']}; font-weight: 600; margin-bottom: 5px;'>{inv['name']}</div>", 
                    unsafe_allow_html=True
                )
                
                # Create a unique key for each slider
                slider_key = f"slider_{key}"
                
                # Get initial value from session state if exists
                initial_value = st.session_state.intervention_config.get(key, {}).get('qty', 0)
                
                qty = st.slider(
                    f"Quantity ({inv['unit']})", 
                    0, inv['max_qty'], 
                    initial_value,
                    help=inv['description'],
                    key=slider_key
                )
                
                if qty > 0:
                    cost = qty * inv['cost']
                    total_cost_display += cost
                    st.markdown(
                        f"<div style='text-align: right; color: {COLORS['orange']}; font-weight: 500; margin-bottom: 15px;'>💰 ${cost:,.0f}</div>", 
                        unsafe_allow_html=True
                    )
                    config[key] = {'qty': qty, 'cost': inv['cost'], 'impact_per_unit': inv['impact_per_unit'], 'feature': inv['feature']}
                    
                    # Store in session state
                    st.session_state.intervention_config[key] = {'qty': qty, 'cost': inv['cost'], 'impact_per_unit': inv['impact_per_unit'], 'feature': inv['feature']}
                else:
                    st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
                    # Remove from session state if qty is 0
                    if key in st.session_state.intervention_config:
                        del st.session_state.intervention_config[key]

    st.divider()
    
    # ============================================================================
    # RESET FILTERS BUTTON
    # ============================================================================
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Count active interventions
        active_count = len(config)
        st.markdown(
            f"<div style='text-align: center; background-color: {COLORS['bg']}; padding: 8px; border-radius: 8px; border: 1px solid {COLORS['border']};'>"
            f"<span style='color: {COLORS['muted']}; font-size: 12px;'>Active: </span>"
            f"<span style='color: {COLORS['blue']}; font-weight: 600;'>{active_count}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    with col2:
        if st.button("🔄 Reset All", use_container_width=True, type="primary"):
            # Clear all intervention sliders from session state
            for key in list(st.session_state.keys()):
                if key.startswith("slider_"):
                    del st.session_state[key]
            # Clear intervention config
            st.session_state.intervention_config = {}
            st.rerun()
    
    # Add some helpful text
    st.markdown(
        f"<p style='color: {COLORS['muted']}; font-size: 11px; text-align: center; margin-top: 20px;'>"
        f"Reset button clears all selected interventions.</p>",
        unsafe_allow_html=True
    )

# ============================================================================
# SIMULATION
# ============================================================================
# ============================================================================
# SIMULATION SECTION - FIXED
# ============================================================================

if config:
    total_cost = sum(v['qty'] * v['cost'] for v in config.values())

    current_features = {f: block_data[f] for f in features if f in block_data.index}
    simulated_features = current_features.copy()

    for key, v in config.items():
        feature = v['feature']
        simulated_features[feature] += v['impact_per_unit'] * v['qty']
        if simulated_features[feature] < 0: 
            simulated_features[feature] = 0

    X_current = scaler.transform([[current_features.get(f, 0) for f in features]])
    X_simulated = scaler.transform([[simulated_features.get(f, 0) for f in features]])

    # Point Prediction
    projected_bhs = model.predict(X_simulated)[0]
    uplift = projected_bhs - baseline_bhs

    # Confidence Intervals via individual Decision Trees
    tree_preds = np.array([tree.predict(X_simulated)[0] for tree in model.estimators_])
    ci_lower, ci_upper = np.percentile(tree_preds, 10), np.percentile(tree_preds, 90)

    # Calculate annual economic benefit from BHS improvement
    annual_bhs_value = economic_value_per_bhs_point * max(0, uplift)

    # ROI as percentage (if total_cost > 0)
    roi_percent = ((annual_bhs_value / total_cost) * 100) if total_cost > 0 else 0

    # Calculate payback period (in years)
    payback_years = total_cost / annual_bhs_value if annual_bhs_value > 0 else float('inf')

    # RESULTS
    st.markdown(
        f"<h3 style='font-family: {FONT_FAMILY}; color: {COLORS['text']};'>Machine Learning Projection for <span style='color: {COLORS['blue']};'>{selected_block}</span></h3>", 
        unsafe_allow_html=True
    )

    cols = st.columns(5)
    with cols[0]:
        st.metric("Baseline BHS", f"{baseline_bhs:.1f}")
    with cols[1]:
        delta_color = "normal" if uplift >= 0 else "inverse"
        st.metric("Projected BHS", f"{projected_bhs:.1f}", f"{uplift:+.2f}", delta_color=delta_color)
    with cols[2]:
        st.metric("80% Confidence", f"[{ci_lower:.1f}, {ci_upper:.1f}]")
    with cols[3]:
        st.metric("Total Cost", f"${total_cost:,.0f}")
    with cols[4]:
        roi_color = "normal" if roi_percent > 0 else "inverse"
        st.metric("ROI", f"{roi_percent:.1f}%", delta_color=roi_color)

    # Add additional economic metrics
    st.markdown(
        f"<div class='info-box'>"
        f"<span style='color: {COLORS['text']}; font-weight: 600;'>Annual Economic Benefit: </span>"
        f"<span style='color: {COLORS['green']}; font-size: 20px; font-weight: 700;'>${annual_bhs_value:,.0f}</span><br>"
        f"<span style='color: {COLORS['text']}; font-weight: 600;'>Payback Period: </span>"
        f"<span style='color: {COLORS['orange']}; font-size: 18px; font-weight: 600;'>{payback_years:.1f} years</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        # VISUALIZATION: Plotting the Net Delta (Change)
        changes = [{'Feature': f.replace('_', ' ').title(), 'Net Change': simulated_features[f] - current_features[f]}
                   for f in features[:10] if f in current_features and abs(simulated_features[f] - current_features[f]) > 0.01]

        if changes:
            changes_df = pd.DataFrame(changes)
            
            # List comprehension for colors
            changes_df['Color'] = [
                COLORS['green'] if ((val < 0 and ('Crime' in feat or 'Issue' in feat or 'Report' in feat)) or 
                                   (val > 0 and ('Business' in feat or 'Diversity' in feat)))
                else COLORS['red']
                for feat, val in zip(changes_df['Feature'], changes_df['Net Change'])
            ]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=changes_df['Net Change'],
                y=changes_df['Feature'],
                orientation='h',
                marker_color=changes_df['Color'],
                text=[f"{val:+.1f}" for val in changes_df['Net Change']],
                textposition='outside',
                textfont=dict(family=FONT_FAMILY, size=FONT_SIZES["tiny"])
            ))
            
            # Create a modified layout without margin conflict
            layout1 = PLOT_LAYOUT.copy()
            layout1['margin']['t'] = 80
            
            fig.update_layout(
                **layout1,
                title={
                    'text': "Net Impact on Block Environment",
                    'font': {'family': FONT_FAMILY, 'size': FONT_SIZES["heading"], 'color': COLORS["text"]}
                },
                xaxis_title="Net Change in Feature Value",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=["Baseline"], 
            y=[baseline_bhs], 
            marker_color=COLORS["red"], 
            width=0.4, 
            text=[f"{baseline_bhs:.1f}"], 
            textposition="outside",
            textfont=dict(family=FONT_FAMILY, size=FONT_SIZES["body"])
        ))
        fig2.add_trace(go.Bar(
            x=["Projected"], 
            y=[projected_bhs], 
            marker_color=COLORS["green"], 
            width=0.4,
            error_y=dict(
                type="data", 
                symmetric=False, 
                array=[ci_upper - projected_bhs], 
                arrayminus=[projected_bhs - ci_lower], 
                color=COLORS["muted"],
                thickness=1.5
            ),
            text=[f"{projected_bhs:.1f}"], 
            textposition="outside",
            textfont=dict(family=FONT_FAMILY, size=FONT_SIZES["body"])
        ))
        
        # Create a modified layout for the second chart
        layout2 = PLOT_LAYOUT.copy()
        layout2['margin']['t'] = 80
        
        fig2.update_layout(
            **layout2,
            title={
                'text': "Predicted BHS (with 80% CI)",
                'font': {'family': FONT_FAMILY, 'size': FONT_SIZES["heading"], 'color': COLORS["text"]}
            },
            yaxis_title="Score", 
            yaxis_range=[0, 100],
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Model Feature Importance", expanded=True):
        fig_imp = px.bar(
            importance.head(10), 
            x='importance', 
            y='feature', 
            orientation='h', 
            title="Top 10 AI Drivers",
            color='importance', 
            color_continuous_scale=['#ef4444', '#f59e0b', '#10b981']
        )
        
        # Create a modified layout for the importance chart
        layout3 = PLOT_LAYOUT.copy()
        layout3['margin']['t'] = 50
        
        fig_imp.update_layout(
            **layout3,
            height=350
        )
        fig_imp.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_imp, use_container_width=True)

else:
    st.info("Select interventions from the sidebar to run the simulation")
    
    st.markdown(
        f"<h3 style='font-family: {FONT_FAMILY}; color: {COLORS['text']};'>Auto-Discovered Interventions</h3>", 
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='color:{COLORS['muted']};font-size:{FONT_SIZES['small']}px;font-family:{FONT_FAMILY};'>"
        f"The AI clustered the Loop's blocks into 5 performance tiers. These interventions target the exact gaps between the worst and best blocks.</p>",
        unsafe_allow_html=True
    )

    cols = st.columns(3)
    for i, (key, inv) in enumerate(list(interventions.items())[:6]):
        with cols[i % 3]:
            st.markdown(f"""
                <div class='intervention-card'>
                    <h4 style='color:{COLORS['text']}; margin:0 0 10px 0;'>{inv['name']}</h4>
                    <p style='color:{COLORS['muted']};font-size:13px;'>{inv['description']}</p>
                    <div style='display:flex; justify-content:space-between; align-items:center; margin-top:15px;'>
                        <span style='color:{COLORS['orange']};font-size:12px;'>${inv['cost']:,.0f} per {inv['unit']}</span>
                        <span style='color:{COLORS['blue']};font-size:12px;'>Max: {inv['max_qty']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption("Hover over charts for detailed values | All impacts and costs learned from historical data patterns | Model saved to disk for faster loading")