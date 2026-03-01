# Loop Pulse — Economic Safety Intelligence Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

Loop Pulse is an economic safety intelligence platform that connects public safety data with economic vitality in Chicago's Loop. The platform quantifies the economic ripple effects of safety issues and identifies where targeted interventions would generate the highest return on investment.

## 📊 Dashboard Suite

### 1. Economic Impact Dashboard (`Visualization.py`)
Interactive visualization of the Loop Pulse feature-engineered dataset, including:
- Monthly crime trends and Business Health Scores
- Crime type distribution analysis
- CTA ridership and 311 request monitoring
- Feature correlations with Business Health Score
- Top crime blocks analysis with mapping

### 2. ML ROI Simulator (`ROI Simulator.py`)
AI-driven intervention ROI simulator that:
- Automatically discovers interventions from data patterns
- Uses Random Forest models with 5-fold cross-validation
- Projects BHS improvements with confidence intervals
- Calculates ROI, payback periods, and annual economic benefits
- Persists trained models for faster loading

### 3. Stakeholder Dashboard (`Stakeholder Dashboard.py`)
Role-based views tailored to different users:
- **City Aldermen**: Policy recommendations & ROI projections
- **Business Owners**: Block trends & competitive positioning
- **Developers**: Investment opportunity zones & growth forecasts

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Your data file: `loop_pulse_features2.csv`

## 🚀 Installation

1. **Clone or download this repository**
   ```bash
   git clone https://github.com/sabalalani/LoopPulse.git
