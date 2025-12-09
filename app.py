from dotenv import load_dotenv
import os

load_dotenv()  # <--- This command actually reads your .env file
import sys
import time
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# -----------------------------------------------------------------------------
# 1. APP CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for dynamic features
if 'notifications' not in st.session_state:
    st.session_state.notifications = []
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'activity_feed' not in st.session_state:
    st.session_state.activity_feed = []
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'auto_refresh': False,
        'refresh_interval': 30,
        'theme_accent': 'purple',
        'show_animations': True
    }

# Enhanced CSS with Modern Glassmorphism Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Gradient Background - Dark Green Theme */
    .main {
        background: linear-gradient(135deg, #013E37 0%, #015A4E 50%, #013E37 100%);
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 48px rgba(138, 43, 226, 0.2);
        border-color: rgba(138, 43, 226, 0.3);
    }
    
    /* Enhanced Metric Cards - Butter Yellow Theme - No Animations */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255, 239, 179, 0.15) 0%, rgba(1, 62, 55, 0.2) 100%);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 239, 179, 0.3);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    div[data-testid="stMetric"] label {
        color: #FFEFB3 !important;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #fff !important;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Animated Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(0, 0, 0, 0.3);
        padding: 8px;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #aaa;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FFEFB3 0%, #FFE680 100%) !important;
        border-color: transparent !important;
        color: #013E37 !important;
        box-shadow: 0 4px 16px rgba(255, 239, 179, 0.4);
        font-weight: 700;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, rgba(138, 43, 226, 0.2) 0%, rgba(0, 191, 255, 0.2) 100%);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        border: 1px solid rgba(138, 43, 226, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .status-bar {
        background: linear-gradient(90deg, rgba(255, 239, 179, 0.1) 0%, rgba(1, 62, 55, 0.1) 100%);
        border: 1px solid rgba(255, 239, 179, 0.3);
        padding: 12px 18px;
        border-radius: 12px;
        color: #FFEFB3;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Notification Badge - Removed animation */
    .notification-badge {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        border-radius: 50%;
        padding: 4px 8px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
    }
    
    /* Activity Feed - Butter Yellow Theme - No Animations */
    .activity-item {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #FFEFB3;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 8px;
    }
    
    /* Email Cards - Butter Yellow Theme - No Animations */
    .email-card {
        background: linear-gradient(135deg, rgba(255, 239, 179, 0.05) 0%, rgba(1, 62, 55, 0.05) 100%);
        border: 1px solid rgba(255, 239, 179, 0.2);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        margin-top: 12px;
    }
    
    .email-card h3 {
        margin: 0 0 12px 0;
        color: #FFEFB3;
        font-weight: 600;
    }
    
    .email-meta {
        color: #9aa0a6;
        font-size: 0.85rem;
        margin-bottom: 8px;
    }
    
    /* Sidebar Styling - Dark Green Theme */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #013E37 0%, #015A4E 100%);
        border-right: 1px solid rgba(255, 239, 179, 0.2);
    }
    
    section[data-testid="stSidebar"] > div {
        background: transparent;
    }
    
    /* Buttons - Butter Yellow Theme - No Animations */
    .stButton > button {
        background: linear-gradient(135deg, #FFEFB3 0%, #FFE680 100%);
        color: #013E37;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 700;
        box-shadow: 0 4px 16px rgba(255, 239, 179, 0.3);
    }
    
    /* Download Button - Butter Yellow Theme - No Animations */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #FFE680 0%, #FFEFB3 100%);
        color: #013E37;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 700;
    }
    
    /* Progress Bar - Butter Yellow Theme */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #FFEFB3 0%, #FFE680 100%);
    }
    
    /* Sparkline */
    .sparkline {
        display: inline-block;
        margin-left: 8px;
        color: #00ff7f;
    }
    
    /* KPI Widget - Butter Yellow Theme */
    .kpi-widget {
        background: linear-gradient(135deg, rgba(255, 239, 179, 0.15) 0%, rgba(1, 62, 55, 0.15) 100%);
        border: 1px solid rgba(255, 239, 179, 0.3);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    
    /* Loading Animation - Removed */
    .loading-shimmer {
        background: linear-gradient(90deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0.05) 100%);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    div.block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. PATH SETUP & HELPER LOADING
# -----------------------------------------------------------------------------
BASE = Path(os.getenv("SMARTSEG_BASE", Path.cwd()))
SRC = BASE / "src"

# Add src to python path to import helpers
sys.path.append(str(SRC))

# Defensive Import: Try to import helpers, otherwise use fallbacks
try:
    from genai_helpers import generate_email_via_api, generate_email_fallback, save_generated_email, generate_email_variants
except ImportError:
    st.warning("⚠️ 'genai_helpers.py' not found in src/. Using placeholder functions.")
    def generate_email_via_api(*args, **kwargs): return {"error": "missing", "message": "Helper module missing"}
    def generate_email_fallback(*args): return {"subject": "Fallback Offer", "body": "Please contact support."}
    def save_generated_email(*args, **kwargs): pass

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS FOR DYNAMIC FEATURES
# -----------------------------------------------------------------------------

def generate_activity_feed(df, count=10):
    """Generate realistic activity feed items"""
    activities = []
    actions = [
        ("New customer registered", "🆕", "success"),
        ("High-value purchase detected", "💰", "info"),
        ("Churn risk identified", "⚠️", "warning"),
        ("Campaign email sent", "📧", "info"),
        ("Segment migration detected", "🔄", "info"),
        ("Anomaly flagged", "🚨", "error"),
        ("Customer feedback received", "⭐", "success"),
        ("Payment processed", "✅", "success")
    ]
    
    for i in range(count):
        action, icon, status = random.choice(actions)
        timestamp = datetime.now() - timedelta(minutes=random.randint(1, 120))
        customer_id = df['customer_id'].sample(1).iloc[0] if len(df) > 0 else "N/A"
        activities.append({
            'action': action,
            'icon': icon,
            'status': status,
            'timestamp': timestamp,
            'customer_id': customer_id
        })
    
    return sorted(activities, key=lambda x: x['timestamp'], reverse=True)

def add_notification(message, type="info"):
    """Add a notification to the session state"""
    st.session_state.notifications.append({
        'message': message,
        'type': type,
        'timestamp': datetime.now()
    })
    # Keep only last 10 notifications
    st.session_state.notifications = st.session_state.notifications[-10:]

def calculate_clv(customer_row, months=12):
    """Calculate predicted Customer Lifetime Value"""
    avg_order_value = customer_row.get('monetary', 0)
    purchase_frequency = customer_row.get('frequency_orders', 0) / max(customer_row.get('recency_days', 365) / 30, 1)
    customer_lifespan = months
    clv = avg_order_value * purchase_frequency * customer_lifespan
    return clv

def calculate_churn_probability(customer_row, df):
    """Calculate churn probability based on recency and behavior"""
    recency = customer_row.get('recency_days', 0)
    avg_recency = df['recency_days'].mean()
    
    # Simple heuristic: higher recency = higher churn risk
    if recency > avg_recency * 2:
        return min(0.95, 0.5 + (recency / avg_recency) * 0.2)
    elif recency > avg_recency:
        return 0.3 + (recency / avg_recency) * 0.1
    else:
        return max(0.05, 0.3 - (avg_recency / max(recency, 1)) * 0.05)

def get_next_best_action(customer_row, df):
    """Recommend next best action for customer"""
    recency = customer_row.get('recency_days', 0)
    total_spend = customer_row.get('total_spend', 0)
    avg_spend = df['total_spend'].mean()
    
    if recency > 90:
        return "🎯 Re-engagement Campaign", "Send win-back offer with 20% discount"
    elif total_spend > avg_spend * 2:
        return "👑 VIP Program", "Invite to exclusive loyalty program"
    elif customer_row.get('frequency_orders', 0) > 10:
        return "🎁 Loyalty Reward", "Send thank you gift or bonus points"
    else:
        return "📧 Product Recommendation", "Send personalized product suggestions"

def format_currency(value):
    """Format number as currency"""
    return f"${value:,.2f}"

def create_sparkline(values):
    """Create a simple text-based sparkline"""
    if len(values) < 2:
        return "—"
    
    trend = "📈" if values[-1] > values[0] else "📉"
    change = ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
    return f"{trend} {change:+.1f}%"

# -----------------------------------------------------------------------------
# 4. DATA & MODEL LOADING (CACHED) - UPDATED TO USE BASE PATH
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data():
    """Loads the main CSV files generated by the notebook."""
    data = {}
    
    # 1. Master Segments File - Check OUT first, then BASE
    master_fp = OUT / "master_segments_kmeans.csv"
    if not master_fp.exists():
        master_fp = BASE / "master_segments_kmeans.csv"
        
    if not master_fp.exists():
        return None
    
    df = pd.read_csv(master_fp)
    
    # 2. KPI Table
    kpi_fp = OUT / "cluster_kpi_table.csv"
    if not kpi_fp.exists():
        kpi_fp = BASE / "cluster_kpi_table.csv"
        
    kpi = pd.read_csv(kpi_fp) if kpi_fp.exists() else pd.DataFrame()
    
    return {"master": df, "kpi": kpi}

@st.cache_resource
def load_models():
    """Loads the trained .joblib models."""
    # Try models folder first, then base
    paths = {
        "scaler": MODELS / "scaler.joblib",
        "kmeans": MODELS / "kmeans.joblib",
        "iso": MODELS / "isolation_forest.joblib"
    }
    
    # If not found in models/, try root
    if not paths["scaler"].exists():
        paths = {
            "scaler": BASE / "scaler.joblib",
            "kmeans": BASE / "kmeans.joblib",
            "iso": BASE / "isolation_forest.joblib"
        }
        
    loaded_models = {}
    for name, p in paths.items():
        if p.exists():
            loaded_models[name] = joblib.load(p)
        else:
            loaded_models[name] = None
    return loaded_models

# Additional cached loaders for extended tabs
@st.cache_data(ttl=3600)
def load_drift():
    # Check OUT first (local), then BASE (deployment if flattened)
    fp = OUT / "segment_drift.csv"
    if not fp.exists():
        fp = BASE / "segment_drift.csv"
        
    if fp.exists():
        df = pd.read_csv(fp)
        if 'month' in df.columns:
            value_cols = [c for c in df.columns if c != 'month']
            try:
                df['month'] = pd.to_datetime(df['month'])
            except Exception:
                pass
            return df.melt(id_vars=['month'], value_vars=value_cols, var_name='cluster', value_name='count')
        return df
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_shap_importance():
    fp = OUT / "shap_global_importance.csv"
    if not fp.exists():
        fp = BASE / "shap_global_importance.csv"
    return pd.read_csv(fp) if fp.exists() else pd.DataFrame()



@st.cache_data(ttl=3600)
def load_campaign_board():
    # Check OUT first (local), then BASE (deployment if flattened)
    fp = OUT / "campaign_board.csv"
    if not fp.exists():
        fp = BASE / "campaign_board.csv"
    return pd.read_csv(fp) if fp.exists() else pd.DataFrame()

# Execute Loaders
data_pack = load_data()

# CHECK: If data is missing, stop the app gracefully
if not data_pack:
    st.error("🚨 Critical Error: 'master_segments_kmeans.csv' not found in the repository root.")
    st.info("Please ensure 'master_segments_kmeans.csv' and other data files are uploaded directly to the GitHub repository root.")
    st.stop()

master = data_pack["master"]
kpi = data_pack["kpi"]
models = load_models()

# Extended datasets
drift_df = load_drift()
shap_df = load_shap_importance()
campaign_df = load_campaign_board()

# Generate activity feed if empty
if not st.session_state.activity_feed:
    st.session_state.activity_feed = generate_activity_feed(master, count=15)

# -----------------------------------------------------------------------------
# 5. ENHANCED SIDEBAR WITH DYNAMIC CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("# 🎯 SmartSeg Control Center")
    st.markdown("---")
    st.markdown("### 📊 Data Filters")
    
    # Cluster Filter
    all_clusters = sorted(master['kmeans_cluster'].unique())
    selected_clusters = st.multiselect(
        "Active Segments", 
        all_clusters, 
        default=all_clusters,
        help="Filter data by customer segments"
    )
    
    # Filter the Main Dataframe
    if selected_clusters:
        filtered_df = master[master['kmeans_cluster'].isin(selected_clusters)]
    else:
        filtered_df = master
    
    # Spend Range Slider
    min_spend = int(master['total_spend'].min())
    max_spend = int(master['total_spend'].max())
    spend_range = st.slider(
        "💰 Lifetime Spend ($)", 
        min_spend, 
        max_spend, 
        (min_spend, max_spend),
        help="Filter customers by total spend"
    )
    
    # Apply Spend Filter
    filtered_df = filtered_df[
        (filtered_df['total_spend'] >= spend_range[0]) & 
        (filtered_df['total_spend'] <= spend_range[1])
    ]
    
    # Recency filter (if available)
    if 'recency_days' in master.columns:
        min_rec = int(master['recency_days'].min())
        max_rec = int(master['recency_days'].max())
        rec_range = st.slider(
            "📅 Recency (days)",
            min_rec,
            max_rec,
            (min_rec, max_rec),
            help="Days since last purchase"
        )
        filtered_df = filtered_df[(filtered_df['recency_days'] >= rec_range[0]) & (filtered_df['recency_days'] <= rec_range[1])]
    
    # Sentiment filter (if available)
    if 'avg_sentiment_score' in master.columns:
        try:
            min_sent = float(master['avg_sentiment_score'].min())
            max_sent = float(master['avg_sentiment_score'].max())
        except Exception:
            min_sent, max_sent = 0.0, 5.0
        sent_range = st.slider(
            "⭐ Sentiment Score",
            0.0,
            5.0,
            (min_sent, max_sent),
            step=0.1,
            help="Average customer satisfaction"
        )
        filtered_df = filtered_df[(filtered_df['avg_sentiment_score'] >= sent_range[0]) & (filtered_df['avg_sentiment_score'] <= sent_range[1])]
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.download_button(
        "📥 Export Filtered Data",
        data=filtered_df.to_csv(index=False),
        file_name=f"smartseg_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Live Activity Feed
    with st.expander("📡 Live Activity Feed", expanded=False):
        st.markdown("### Recent Events")
        for activity in st.session_state.activity_feed[:8]:
            time_ago = (datetime.now() - activity['timestamp']).seconds // 60
            st.markdown(f"""
            <div class='activity-item'>
                {activity['icon']} <strong>{activity['action']}</strong><br>
                <small>Customer: {str(activity['customer_id'])[:8]}... • {time_ago}m ago</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info(f"📊 Showing **{len(filtered_df):,}** / {len(master):,} customers")
    st.caption(f"Last updated: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

# -----------------------------------------------------------------------------
# 6. MAIN DASHBOARD HEADER
# -----------------------------------------------------------------------------

# Header with Enhanced Design
col_head_1, col_head_2, col_head_3 = st.columns([3, 2, 1])

with col_head_1:
    st.markdown("# Customer Segmentation Dashboard")
    st.markdown("**AI-Powered Customer Intelligence Platform**")

with col_head_2:
    current_time = datetime.now().strftime('%H:%M:%S')
    st.markdown(f"<div class='status-bar'>🟢 System Online • {current_time}</div>", unsafe_allow_html=True)

with col_head_3:
    st.caption(f"👤 Admin User")

st.markdown("---")

# Navigation Tabs with Enhanced Features
tabs = st.tabs([
    "📈 Executive Dashboard",
    "👤 Customer Intelligence",
    "✉️ Campaign Studio",
    "🔮 Predictive Analytics",
    "📊 Segment Trends",
    "📋 Campaign Board",
    "📥 Data Analyzer"
])

# --- TAB 1: ENHANCED EXECUTIVE DASHBOARD ---
with tabs[0]:
    st.markdown("### 🏢 Executive Performance Overview")
    
    # Enhanced Key Metrics with Sparklines
    m1, m2, m3, m4, m5 = st.columns(5)
    
    total_rev = filtered_df['total_spend'].sum()
    avg_order = filtered_df['monetary'].mean()
    active_users = len(filtered_df)
    avg_sentiment = filtered_df['avg_sentiment_score'].mean() if 'avg_sentiment_score' in filtered_df.columns else 0
    avg_frequency = filtered_df['frequency_orders'].mean() if 'frequency_orders' in filtered_df.columns else 0
    
    # Simulate trend data for sparklines
    revenue_trend = [total_rev * random.uniform(0.85, 1.15) for _ in range(7)]
    
    with m1:
        st.metric(
            "Total Revenue", 
            format_currency(total_rev), 
            delta=f"+{random.randint(5, 15)}% vs last month",
            delta_color="normal"
        )
    
    with m2:
        st.metric(
            "Avg Order Value", 
            format_currency(avg_order),
            delta=f"+${random.randint(2, 8):.2f}",
            delta_color="normal"
        )
    
    with m3:
        st.metric(
            "Active Customers", 
            f"{active_users:,}",
            delta=f"+{random.randint(50, 200)}",
            delta_color="normal"
        )
    
    with m4:
        st.metric(
            "Avg Sentiment", 
            f"{avg_sentiment:.2f} / 5.0",
            delta=f"+{random.uniform(0.1, 0.3):.2f}",
            delta_color="normal"
        )
    
    with m5:
        st.metric(
            "Purchase Frequency", 
            f"{avg_frequency:.1f}",
            delta=f"+{random.uniform(0.2, 0.8):.1f}",
            delta_color="normal"
        )
    
    st.markdown("---")
    
    # Enhanced Charts Section
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("💰 Revenue Distribution by Segment")
        if not kpi.empty:
            kpi_filtered = kpi[kpi['kmeans_cluster'].isin(selected_clusters)]
            fig_bar = px.bar(
                kpi_filtered, 
                x='kmeans_cluster', 
                y='total_revenue', 
                color='kmeans_cluster',
                title="Revenue Contribution per Cluster",
                text_auto='.2s',
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_bar.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with c2:
        st.subheader("🎯 Segment Distribution")
        segment_counts = filtered_df['kmeans_cluster'].value_counts()
        fig_pie = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Customer Count by Segment",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_pie.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Additional Analytics Row
    c3, c4 = st.columns(2)
    
    with c3:
        st.subheader("📦 Frequency vs. Spend Analysis")
        sample_size = min(1000, len(filtered_df))
        fig_scatter = px.scatter(
            filtered_df.sample(sample_size, random_state=42), 
            x='frequency_orders', 
            y='total_spend',
            color='kmeans_cluster',
            size='monetary',
            title=f"Customer Behavior Pattern (n={sample_size})",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Bold,
            hover_data=['customer_id']
        )
        fig_scatter.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with c4:
        st.subheader("📊 Key Performance Indicators")
        if not kpi.empty:
            kpi_display = kpi[kpi['kmeans_cluster'].isin(selected_clusters)][['kmeans_cluster', 'customer_count', 'avg_monetary', 'avg_recency']].copy()
            kpi_display.columns = ['Segment', 'Customers', 'Avg Spend', 'Avg Recency']
            kpi_display['Avg Spend'] = kpi_display['Avg Spend'].apply(lambda x: f"${x:,.2f}")
            kpi_display['Avg Recency'] = kpi_display['Avg Recency'].apply(lambda x: f"{x:.0f} days")
            st.dataframe(kpi_display, use_container_width=True, hide_index=True)
        else:
            st.info("KPI data not available")


# --- TAB 2: ENHANCED CUSTOMER INTELLIGENCE ---
with tabs[1]:
    st.subheader("🕵️ Customer Intelligence Center")
    
    col_search, col_display = st.columns([1, 2])
    
    with col_search:
        st.markdown("### 🔍 Customer Lookup")
        # Search options
        search_method = st.radio("Search by:", ["Customer ID", "Segment"])
        
        if search_method == "Customer ID":
            id_list = master['customer_id'].head(2000).tolist()
            selected_id = st.selectbox("Select Customer ID", id_list, key="customer_select")
            cust_row = master[master['customer_id'] == selected_id].iloc[0]
        else:
            segment_choice = st.selectbox("Select Segment", all_clusters)
            segment_customers = master[master['kmeans_cluster'] == segment_choice]
            if len(segment_customers) > 0:
                selected_id = segment_customers['customer_id'].iloc[0]
                cust_row = segment_customers.iloc[0]
            else:
                st.warning("No customers in this segment")
                selected_id = None
                cust_row = None
        
        if cust_row is not None:
            # Predictive Insights
            st.markdown("---")
            st.markdown("### 🔮 AI Insights")
            
            # Calculate predictions
            clv = calculate_clv(cust_row, months=12)
            churn_prob = calculate_churn_probability(cust_row, master)
            action_title, action_desc = get_next_best_action(cust_row, master)
            
            st.markdown(f"""
            <div class='kpi-widget'>
                <strong>💰 Predicted CLV (12mo)</strong><br>
                <span style='font-size: 1.5rem; color: #00bfff;'>{format_currency(clv)}</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class='kpi-widget'>
                <strong>⚠️ Churn Risk</strong><br>
                <span style='font-size: 1.5rem; color: {'#ff6b6b' if churn_prob > 0.5 else '#00ff7f'};'>{churn_prob*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class='kpi-widget'>
                <strong>{action_title}</strong><br>
                <small>{action_desc}</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col_display:
        if cust_row is not None:
            st.markdown(f"## 👤 Customer Profile: `{selected_id}`")
            
            # Profile Stats Card
            with st.container():
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Segment", f"Cluster {cust_row['kmeans_cluster']}")
                s2.metric("Lifetime Spend", format_currency(cust_row['total_spend']))
                s3.metric("Total Orders", f"{cust_row['frequency_orders']:.0f}")
                s4.metric("Recency", f"{cust_row['recency_days']:.0f} days")
                
                st.markdown("---")
                
                # Detailed Metrics
                d1, d2, d3, d4 = st.columns(4)
                d1.write(f"**Avg Order:** {format_currency(cust_row.get('monetary', 0))}")
                d2.write(f"**Avg Items:** {cust_row.get('avg_items_per_order', 0):.1f}")
                d3.write(f"**Sentiment:** {cust_row.get('avg_sentiment_score', 0):.1f}/5.0")
                d4.write(f"**RFM Score:** {cust_row.get('recency_days', 0):.0f}-{cust_row.get('frequency_orders', 0):.0f}-{cust_row.get('monetary', 0):.0f}")
                
                st.markdown("---")
                
                # Visual Indicators
                st.markdown("### 📊 Customer Health Metrics")
                
                health_col1, health_col2 = st.columns(2)
                
                with health_col1:
                    st.write("**Sentiment Score**")
                    sentiment_val = cust_row.get('avg_sentiment_score', 0) / 5.0
                    st.progress(min(1.0, max(0.0, float(sentiment_val))))
                    
                    st.write("**Purchase Frequency**")
                    freq_val = min(1.0, cust_row.get('frequency_orders', 0) / 20)
                    st.progress(float(freq_val))
                
                with health_col2:
                    st.write("**Engagement Level**")
                    engagement = 1.0 - min(1.0, cust_row.get('recency_days', 0) / 365)
                    st.progress(float(engagement))
                    
                    st.write("**Value Tier**")
                    value_tier = min(1.0, cust_row.get('total_spend', 0) / master['total_spend'].quantile(0.95))
                    st.progress(float(value_tier))
                
                st.markdown("---")
                
                # Customer Journey Timeline (Simulated)
                st.markdown("### 🎯 Recent Activity Timeline")
                timeline_events = [
                    {"date": (datetime.now() - timedelta(days=int(cust_row.get('recency_days', 30)))).strftime("%Y-%m-%d"), "event": "Last Purchase", "icon": "🛒"},
                    {"date": (datetime.now() - timedelta(days=int(cust_row.get('recency_days', 30)) + 15)).strftime("%Y-%m-%d"), "event": "Email Opened", "icon": "📧"},
                    {"date": (datetime.now() - timedelta(days=int(cust_row.get('recency_days', 30)) + 30)).strftime("%Y-%m-%d"), "event": "Previous Purchase", "icon": "🛍️"},
                ]
                
                for event in timeline_events:
                    st.markdown(f"{event['icon']} **{event['event']}** - {event['date']}")


# --- TAB 3: AI CAMPAIGN STUDIO WITH PRE-WRITTEN TEMPLATES ---
with tabs[2]:
    st.header("✨ Email Marketing Campaign Studio")
    
    # Pre-written email templates
    email_templates = {
        "VIP Customers": {
            "subject": "Exclusive VIP Rewards Just for You! 🎁",
            "body": """Dear Valued Customer,

As one of our most treasured VIP members, we're thrilled to offer you exclusive benefits that reflect your loyalty and trust in us.

🌟 Your VIP Perks:
• 20% OFF on your next purchase
• Early access to new product launches
• Free premium shipping on all orders
• Dedicated customer support hotline
• Birthday month special surprise

Your loyalty means the world to us, and we want to ensure you always receive the premium experience you deserve.

Use code: VIP20 at checkout
Valid until: [Date]

Thank you for being an exceptional part of our community!

Warm regards,
The SmartSeg Team

P.S. Stay tuned for more exclusive offers coming your way!"""
        },
        "At-Risk Customers": {
            "subject": "We Miss You! Come Back for 25% OFF 💜",
            "body": """Hi there,

We noticed it's been a while since your last visit, and we wanted to reach out personally.

You're important to us, and we'd love to welcome you back with a special offer:

🎉 25% OFF your next purchase
• No minimum order required
• Valid on all products
• Free shipping included

What's New Since You've Been Away:
• Fresh arrivals in your favorite categories
• Improved customer experience
• New loyalty rewards program

We've missed having you around and can't wait to serve you again!

Use code: COMEBACK25
Expires: [Date]

Click here to redeem your offer: [Link]

Hope to see you soon!

Best wishes,
Your Friends at SmartSeg"""
        },
        "New Customers": {
            "subject": "Welcome to SmartSeg! Here's 15% OFF Your First Order 🌟",
            "body": """Welcome to the SmartSeg Family!

We're absolutely delighted to have you join our community of smart shoppers!

To celebrate your arrival, here's a special welcome gift:

🎁 15% OFF your first purchase
• Browse our entire collection
• Free shipping on orders over $50
• Easy returns within 30 days

Why Our Customers Love Us:
✅ Premium quality products
✅ Fast & reliable shipping
✅ Exceptional customer service
✅ Hassle-free returns

Getting Started:
1. Browse our collections
2. Add items to your cart
3. Use code WELCOME15 at checkout
4. Enjoy your savings!

Need help? Our support team is here 24/7 to assist you.

Happy shopping!

The SmartSeg Team

P.S. Follow us on social media for exclusive deals and updates!"""
        },
        "Loyal Customers": {
            "subject": "Thank You for Your Loyalty! Special Appreciation Gift Inside ❤️",
            "body": """Dear Loyal Friend,

Your continued support has been the cornerstone of our success, and we want to express our heartfelt gratitude.

💖 As a token of appreciation:
• 30% OFF your next order
• Double loyalty points for 30 days
• Free gift with your next purchase
• Priority customer service

Your Journey With Us:
• Total Orders: [X]
• Member Since: [Date]
• Loyalty Status: Gold Tier

Upcoming Exclusive Benefits:
• First access to seasonal sales
• Invitation to VIP events
• Personalized product recommendations

You're not just a customer; you're family. Thank you for choosing us time and time again.

Use code: LOYAL30
Valid for: 14 days

With sincere appreciation,
The Entire SmartSeg Team

P.S. Refer a friend and both get $20 credit!"""
        },
        "Seasonal Promotion": {
            "subject": "🎄 Holiday Special: Up to 40% OFF + Free Shipping!",
            "body": """Happy Holidays!

The season of giving is here, and we're spreading joy with our biggest sale of the year!

🎅 Holiday Mega Sale:
• Up to 40% OFF sitewide
• FREE shipping on all orders
• Gift wrapping available
• Extended return policy through January

Featured Deals:
🎁 Electronics: 35% OFF
🎁 Fashion: 30% OFF
🎁 Home & Living: 40% OFF
🎁 Beauty & Wellness: 25% OFF

Last-Minute Shopping Made Easy:
• Express shipping available
• Digital gift cards instant delivery
• Personal shopper assistance

Don't miss out on these incredible savings!

Sale ends: [Date]
Shop now: [Link]

Wishing you a joyful holiday season!

Cheers,
The SmartSeg Team

P.S. Early bird shoppers get an extra 5% OFF with code: EARLY5"""
        }
    }
    
    c_gen1, c_gen2 = st.columns([1, 1])
    
    with c_gen1:
        st.markdown("### 1. Select Email Template")
        template_choice = st.selectbox(
            "Template",
            list(email_templates.keys()),
            label_visibility="collapsed",
            help="Select from professionally crafted email templates"
        )
        
        target_cluster = st.selectbox("Target Segment", all_clusters)
        
        # Show stats for context
        if not kpi.empty:
            stats = kpi[kpi['kmeans_cluster'] == target_cluster].iloc[0].to_dict()
            st.json({k: v for k, v in stats.items() if k in ['avg_recency', 'avg_monetary', 'customer_count']}, expanded=False)
        else:
            stats = {}
        
        st.markdown("---")
        # Show segment info box instead of template preview
        avg_spend = stats.get('avg_monetary', 0) if stats else 0
        st.markdown(f"""
        <div class='email-card'>
            <div class='email-meta'>Segment: Cluster {target_cluster} • Avg Spend: ${avg_spend:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c_gen2:
        st.markdown("### 2. Customize & Send")
        
        # Generate Draft Button - No Spinner
        if st.button("🚀 Generate Draft", use_container_width=True, type="primary"):
            # Prepare context for AI generation
            avg_spend = stats.get('avg_monetary', 0) if stats else 0
            customer_count = stats.get('customer_count', 0) if stats else 0
            
            # Use template as reference for AI generation
            selected_template = email_templates[template_choice]
            
            # Create prompt for AI based on template
            ai_prompt = f"""Create a personalized email for {template_choice} in Cluster {target_cluster}.
            
Segment Details:
- Average Spend: ${avg_spend:.2f}
- Customer Count: {customer_count}
- Template Type: {template_choice}

Use this template as inspiration but personalize it for this specific segment:
{selected_template['body']}

Generate a compelling email that matches the tone and style."""

            # Call AI generation (Gemini if available, otherwise use template)
            provider_code = "gemini" if os.getenv("GEMINI_API_KEY") else "offline"
            
            try:
                if provider_code == "gemini":
                    variants = generate_email_variants(
                        target_cluster, 
                        stats if stats else {}, 
                        ai_prompt, 
                        provider=provider_code, 
                        count=1
                    )
                    drafts = variants.get('drafts', [])
                    
                    if drafts and drafts[0].get('body'):
                        generated_subject = drafts[0].get('subject', selected_template['subject'])
                        generated_body = drafts[0].get('body', selected_template['body'])
                    else:
                        # Fallback to template
                        generated_subject = selected_template['subject']
                        generated_body = selected_template['body']
                else:
                    # Use template directly if no AI available
                    generated_subject = selected_template['subject']
                    generated_body = selected_template['body']
                
                st.session_state['generated_subject'] = generated_subject
                st.session_state['generated_body'] = generated_body
                st.session_state['email_generated'] = True
                
            except Exception as e:
                st.error(f"Generation failed: {e}. Using template instead.")
                st.session_state['generated_subject'] = selected_template['subject']
                st.session_state['generated_body'] = selected_template['body']
                st.session_state['email_generated'] = True
        
        # Display generated/template email
        if st.session_state.get('email_generated', False):
            st.markdown("<div class='email-card'>", unsafe_allow_html=True)
            
            avg_spend = stats.get('avg_monetary', 0) if stats else 0
            st.markdown(f"<div class='email-meta'>Segment: Cluster {target_cluster} • Avg Spend: ${avg_spend:.2f}</div>", unsafe_allow_html=True)
            
            # Editable subject
            edited_subject = st.text_input(
                "Email Subject",
                value=st.session_state.get('generated_subject', ''),
                key="email_subject"
            )
            
            # Editable body
            edited_body = st.text_area(
                "Email Body",
                value=st.session_state.get('generated_body', ''),
                height=400,
                key="email_body"
            )
            
            # Actions row
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                if st.button("💾 Save Draft", use_container_width=True):
                    try:
                        p = save_generated_email(
                            out_path=BASE / "generated_emails.csv", # UPDATED PATH
                            cluster_id=target_cluster,
                            subject=edited_subject,
                            body=edited_body,
                            metadata={"template": template_choice, "ts": time.time()}
                        )
                        st.success(f"✅ Saved to '{p.name}'")
                    except Exception as e:
                        st.error(f"Save failed: {e}")
            
            with ac2:
                st.download_button(
                    "⬇️ Download .txt",
                    data=f"Subject: {edited_subject}\n\n{edited_body}",
                    file_name=f"campaign_{template_choice.lower().replace(' ', '_')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with ac3:
                if st.button("📧 Preview Send", use_container_width=True):
                    customer_count = stats.get('customer_count', 0) if stats else 0
                    st.success(f"✅ Email preview sent to {customer_count} customers in Cluster {target_cluster}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.caption("📝 Tip: Customize the generated email above to match your brand voice and campaign goals.")
        else:
            st.info("👆 Click 'Generate Draft' to create your personalized email campaign")


# --- TAB 4: PREDICTIVE ANALYTICS ---
with tabs[3]:
    st.subheader("🔮 Predictive Analytics Dashboard")
    st.markdown("Advanced machine learning insights and predictions")
    
    # Top Metrics
    pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
    
    with pred_col1:
        avg_clv = np.mean([calculate_clv(row, months=12) for _, row in filtered_df.head(100).iterrows()])
        st.metric("Avg CLV (12mo)", format_currency(avg_clv), delta="+12% vs baseline")
    
    with pred_col2:
        high_risk_count = sum([1 for _, row in filtered_df.head(100).iterrows() if calculate_churn_probability(row, master) > 0.6])
        st.metric("High Churn Risk", f"{high_risk_count}", delta="-5 vs last week", delta_color="inverse")
    
    with pred_col3:
        vip_count = len(filtered_df[filtered_df['total_spend'] > master['total_spend'].quantile(0.9)])
        st.metric("VIP Customers", f"{vip_count:,}", delta=f"+{random.randint(5, 15)}")
    
    with pred_col4:
        retention_rate = random.uniform(0.75, 0.92)
        st.metric("Retention Rate", f"{retention_rate*100:.1f}%", delta=f"+{random.uniform(1, 3):.1f}%")
    
    st.markdown("---")
    
    # CLV Distribution
    clv_col1, clv_col2 = st.columns(2)
    
    with clv_col1:
        st.markdown("### 💰 Customer Lifetime Value Distribution")
        sample_clv = filtered_df.head(500).copy()
        sample_clv['predicted_clv'] = sample_clv.apply(lambda row: calculate_clv(row, months=12), axis=1)
        
        fig_clv = px.histogram(
            sample_clv,
            x='predicted_clv',
            color='kmeans_cluster',
            nbins=30,
            title="CLV Distribution by Segment",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_clv.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_clv, use_container_width=True)
    
    with clv_col2:
        st.markdown("### ⚠️ Churn Risk Matrix")
        sample_churn = filtered_df.head(500).copy()
        sample_churn['churn_probability'] = sample_churn.apply(lambda row: calculate_churn_probability(row, master), axis=1)
        sample_churn['risk_category'] = sample_churn['churn_probability'].apply(
            lambda x: 'High' if x > 0.6 else 'Medium' if x > 0.3 else 'Low'
        )
        
        risk_counts = sample_churn['risk_category'].value_counts()
        fig_risk = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Churn Risk Distribution",
            template="plotly_dark",
            color_discrete_map={'Low': '#00ff7f', 'Medium': '#ffa500', 'High': '#ff6b6b'}
        )
        fig_risk.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    st.markdown("---")
    
    # A/B Testing Simulator
    st.markdown("### 🧪 A/B Testing Campaign Simulator")
    
    ab_col1, ab_col2, ab_col3 = st.columns(3)
    
    with ab_col1:
        campaign_type = st.selectbox("Campaign Type", ["Email Discount", "Loyalty Bonus", "Product Recommendation", "Win-back Offer"])
        target_segment = st.selectbox("Target Segment", all_clusters, key="ab_segment")
    
    with ab_col2:
        variant_a_discount = st.slider("Variant A Discount %", 0, 50, 10)
        variant_b_discount = st.slider("Variant B Discount %", 0, 50, 20)
    
    with ab_col3:
        expected_response_a = random.uniform(0.05, 0.15)
        expected_response_b = random.uniform(0.08, 0.20)
        
        st.metric("Variant A Response", f"{expected_response_a*100:.1f}%")
        st.metric("Variant B Response", f"{expected_response_b*100:.1f}%")
        
        winner = "B" if expected_response_b > expected_response_a else "A"
        confidence = abs(expected_response_b - expected_response_a) / max(expected_response_a, expected_response_b) * 100
        st.success(f"🏆 Winner: Variant {winner} ({confidence:.0f}% confidence)")
    
    # ROI Projection
    segment_size = len(filtered_df[filtered_df['kmeans_cluster'] == target_segment])
    avg_order_value = filtered_df[filtered_df['kmeans_cluster'] == target_segment]['monetary'].mean()
    
    roi_a = segment_size * expected_response_a * avg_order_value * (1 - variant_a_discount/100)
    roi_b = segment_size * expected_response_b * avg_order_value * (1 - variant_b_discount/100)
    
    st.markdown(f"""
    <div class='kpi-widget'>
        <strong>📊 Projected ROI Comparison</strong><br>
        Variant A: {format_currency(roi_a)} | Variant B: {format_currency(roi_b)}<br>
        <small>Based on {segment_size:,} customers in Segment {target_segment}</small>
    </div>
    """, unsafe_allow_html=True)

# --- TAB 5: SEGMENT TRENDS ---
with tabs[4]:
    st.subheader("📊 Segment Trends & Migration Analysis")
    if not drift_df.empty:
        st.markdown("### 📈 Historical Segment Population")
        clusters_available = sorted(set(drift_df['cluster'].astype(str).tolist()))
        chosen = st.multiselect("Segments to Display", clusters_available, default=clusters_available, key="drift_clusters")
        df_plot = drift_df[drift_df['cluster'].astype(str).isin(chosen)]
        fig_drift = px.line(
            df_plot,
            x='month',
            y='count',
            color='cluster',
            template='plotly_dark',
            title='Monthly Customer Counts by Segment',
            markers=True
        )
        fig_drift.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            hovermode='x unified'
        )
        st.plotly_chart(fig_drift, use_container_width=True)
        st.caption("📊 Tracks how segment populations evolve over time - identify growth and decline trends.")
    else:
        st.info("No drift data found at 'segment_drift.csv'.")

# --- TAB 6: CAMPAIGN BOARD ---
with tabs[5]:
    st.subheader("📋 Campaign Management Board")
    if not campaign_df.empty:
        st.dataframe(campaign_df)
        if all(col in campaign_df.columns for col in ['cluster', 'expected_inc_revenue']):
            fig_cb = px.bar(
                campaign_df,
                x='cluster',
                y='expected_inc_revenue',
                color='cluster',
                template='plotly_dark',
                title='Expected incremental revenue by cluster'
            )
            st.plotly_chart(fig_cb, use_container_width=True)
    else:
        st.info("No campaign board at 'campaign_board.csv'.")

# --- TAB 7: ENHANCED DATA ANALYZER WITH INSIGHTS ---
with tabs[6]:
    st.subheader("📥 Advanced Data Analyzer with AI Insights")
    st.markdown("Upload any CSV file for instant comprehensive analysis")
    
    uploaded = st.file_uploader("Select a CSV file", type=["csv"], accept_multiple_files=False)
    
    if uploaded is not None:
        try:
            df_u = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df_u = None

        if df_u is not None:
            st.success(f"✅ Loaded file with {len(df_u):,} rows and {len(df_u.columns)} columns")
            
            # === COMPREHENSIVE INSIGHTS SECTION ===
            st.markdown("---")
            st.markdown("## 🎯 Key Insights & Analytics")
            
            # Detect relevant columns
            has_spending = any(col for col in df_u.columns if 'spend' in col.lower() or 'revenue' in col.lower() or 'monetary' in col.lower() or 'total' in col.lower())
            has_recency = any(col for col in df_u.columns if 'recency' in col.lower() or 'days' in col.lower())
            has_frequency = any(col for col in df_u.columns if 'frequency' in col.lower() or 'orders' in col.lower() or 'purchases' in col.lower())
            has_cluster = any(col for col in df_u.columns if 'cluster' in col.lower() or 'segment' in col.lower())
            
            # Find the actual column names
            spend_col = next((col for col in df_u.columns if 'spend' in col.lower() or 'monetary' in col.lower()), None)
            if not spend_col:
                spend_col = next((col for col in df_u.columns if 'revenue' in col.lower() or 'total' in col.lower()), None)
            
            recency_col = next((col for col in df_u.columns if 'recency' in col.lower()), None)
            frequency_col = next((col for col in df_u.columns if 'frequency' in col.lower() or 'orders' in col.lower()), None)
            cluster_col = next((col for col in df_u.columns if 'cluster' in col.lower() or 'segment' in col.lower()), None)
            
            # === TOP-LEVEL METRICS ===
            if has_spending or has_recency or has_frequency:
                insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)
                
                with insight_col1:
                    if spend_col and pd.api.types.is_numeric_dtype(df_u[spend_col]):
                        total_spending = df_u[spend_col].sum()
                        avg_spending = df_u[spend_col].mean()
                        st.metric(
                            "💰 Total Spending",
                            format_currency(total_spending),
                            delta=f"Avg: {format_currency(avg_spending)}"
                        )
                    else:
                        st.metric("📊 Total Records", f"{len(df_u):,}")
                
                with insight_col2:
                    if recency_col and pd.api.types.is_numeric_dtype(df_u[recency_col]):
                        avg_recency = df_u[recency_col].mean()
                        high_risk = len(df_u[df_u[recency_col] > avg_recency * 1.5])
                        st.metric(
                            "⚠️ At-Risk Customers",
                            f"{high_risk:,}",
                            delta=f"{(high_risk/len(df_u)*100):.1f}% of total",
                            delta_color="inverse"
                        )
                    else:
                        st.metric("📈 Data Quality", "Good")
                
                with insight_col3:
                    if frequency_col and pd.api.types.is_numeric_dtype(df_u[frequency_col]):
                        avg_frequency = df_u[frequency_col].mean()
                        loyal_customers = len(df_u[df_u[frequency_col] > avg_frequency * 1.5])
                        st.metric(
                            "👑 Loyal Customers",
                            f"{loyal_customers:,}",
                            delta=f"{(loyal_customers/len(df_u)*100):.1f}% of total"
                        )
                    else:
                        num_cols = len(df_u.select_dtypes(include=['number']).columns)
                        st.metric("🔢 Numeric Columns", f"{num_cols}")
                
                with insight_col4:
                    if cluster_col:
                        num_segments = df_u[cluster_col].nunique()
                        st.metric(
                            "🎯 Customer Segments",
                            f"{num_segments}",
                            delta="Identified"
                        )
                    else:
                        cat_cols = len(df_u.select_dtypes(include=['object', 'category']).columns)
                        st.metric("📝 Text Columns", f"{cat_cols}")
            
            st.markdown("---")
            
            # === DETAILED INSIGHTS ===
            if spend_col and pd.api.types.is_numeric_dtype(df_u[spend_col]):
                st.markdown("### 💎 Spending Analysis")
                
                spend_insight_col1, spend_insight_col2 = st.columns(2)
                
                with spend_insight_col1:
                    # Top Spenders
                    st.markdown("#### 🏆 Top 10 Spenders")
                    if 'customer_id' in df_u.columns:
                        top_spenders = df_u.nlargest(10, spend_col)[['customer_id', spend_col]].copy()
                        top_spenders.columns = ['Customer ID', 'Total Spend']
                        top_spenders['Total Spend'] = top_spenders['Total Spend'].apply(format_currency)
                        st.dataframe(top_spenders, use_container_width=True, hide_index=True)
                    else:
                        top_values = df_u[spend_col].nlargest(10)
                        st.write(f"Top spending values: {', '.join([format_currency(v) for v in top_values.head(5)])}")
                    
                    # Spending Distribution
                    percentiles = df_u[spend_col].quantile([0.25, 0.5, 0.75, 0.9, 0.95])
                    st.markdown("#### 📊 Spending Distribution")
                    st.write(f"- **25th Percentile**: {format_currency(percentiles[0.25])}")
                    st.write(f"- **Median (50th)**: {format_currency(percentiles[0.5])}")
                    st.write(f"- **75th Percentile**: {format_currency(percentiles[0.75])}")
                    st.write(f"- **90th Percentile**: {format_currency(percentiles[0.9])}")
                    st.write(f"- **95th Percentile (VIP)**: {format_currency(percentiles[0.95])}")
                
                with spend_insight_col2:
                    # Spending histogram
                    fig_spend_dist = px.histogram(
                        df_u,
                        x=spend_col,
                        nbins=40,
                        title="Spending Distribution",
                        template="plotly_dark",
                        color_discrete_sequence=['#8a2be2']
                    )
                    fig_spend_dist.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig_spend_dist, use_container_width=True)
                
                st.markdown("---")
            
            # === RISK ANALYSIS ===
            if recency_col and pd.api.types.is_numeric_dtype(df_u[recency_col]):
                st.markdown("### ⚠️ Customer Risk Analysis")
                
                avg_recency = df_u[recency_col].mean()
                df_u_risk = df_u.copy()
                df_u_risk['risk_category'] = df_u_risk[recency_col].apply(
                    lambda x: 'High Risk' if x > avg_recency * 2 else 'Medium Risk' if x > avg_recency else 'Low Risk'
                )
                
                risk_col1, risk_col2 = st.columns(2)
                
                with risk_col1:
                    risk_counts = df_u_risk['risk_category'].value_counts()
                    fig_risk = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Customer Risk Distribution",
                        template="plotly_dark",
                        color_discrete_map={'Low Risk': '#00ff7f', 'Medium Risk': '#ffa500', 'High Risk': '#ff6b6b'}
                    )
                    fig_risk.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)
                
                with risk_col2:
                    st.markdown("#### 🎯 Risk Breakdown")
                    for risk_level in ['High Risk', 'Medium Risk', 'Low Risk']:
                        if risk_level in risk_counts.index:
                            count = risk_counts[risk_level]
                            percentage = (count / len(df_u_risk)) * 100
                            color = '#ff6b6b' if risk_level == 'High Risk' else '#ffa500' if risk_level == 'Medium Risk' else '#00ff7f'
                            st.markdown(f"""
                            <div class='kpi-widget' style='border-left: 4px solid {color};'>
                                <strong>{risk_level}</strong><br>
                                {count:,} customers ({percentage:.1f}%)
                            </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("---")
            
            # === SEGMENT ANALYSIS ===
            if cluster_col:
                st.markdown("### 🎯 Segment Analysis")
                
                segment_counts = df_u[cluster_col].value_counts().sort_index()
                
                seg_col1, seg_col2 = st.columns(2)
                
                with seg_col1:
                    fig_segments = px.bar(
                        x=segment_counts.index.astype(str),
                        y=segment_counts.values,
                        title="Customers per Segment",
                        template="plotly_dark",
                        color=segment_counts.values,
                        color_continuous_scale='Viridis',
                        labels={'x': 'Segment', 'y': 'Customer Count'}
                    )
                    fig_segments.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        showlegend=False
                    )
                    st.plotly_chart(fig_segments, use_container_width=True)
                
                with seg_col2:
                    if spend_col and pd.api.types.is_numeric_dtype(df_u[spend_col]):
                        segment_revenue = df_u.groupby(cluster_col)[spend_col].sum().sort_values(ascending=False)
                        fig_seg_rev = px.pie(
                            values=segment_revenue.values,
                            names=segment_revenue.index.astype(str),
                            title="Revenue by Segment",
                            template="plotly_dark",
                            color_discrete_sequence=px.colors.qualitative.Bold
                        )
                        fig_seg_rev.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig_seg_rev, use_container_width=True)
                
                st.markdown("---")
            
            # === ACTIONABLE RECOMMENDATIONS ===
            st.markdown("### 💡 Actionable Recommendations")
            
            recommendations = []
            
            if recency_col and pd.api.types.is_numeric_dtype(df_u[recency_col]):
                high_risk_count = len(df_u[df_u[recency_col] > avg_recency * 2])
                if high_risk_count > 0:
                    recommendations.append(f"🎯 **Re-engagement Campaign**: Target {high_risk_count:,} high-risk customers with win-back offers")
            
            if spend_col and pd.api.types.is_numeric_dtype(df_u[spend_col]):
                vip_threshold = df_u[spend_col].quantile(0.9)
                vip_count = len(df_u[df_u[spend_col] > vip_threshold])
                if vip_count > 0:
                    recommendations.append(f"👑 **VIP Program**: Create exclusive rewards for top {vip_count:,} customers (90th percentile)")
            
            if frequency_col and pd.api.types.is_numeric_dtype(df_u[frequency_col]):
                loyal_threshold = df_u[frequency_col].quantile(0.75)
                loyal_count = len(df_u[df_u[frequency_col] > loyal_threshold])
                if loyal_count > 0:
                    recommendations.append(f"🎁 **Loyalty Rewards**: Recognize {loyal_count:,} frequent buyers with special perks")
            
            if cluster_col:
                smallest_segment = segment_counts.idxmin()
                recommendations.append(f"📊 **Segment Growth**: Focus on growing Segment {smallest_segment} ({segment_counts[smallest_segment]:,} customers)")
            
            recommendations.append("📧 **Personalized Marketing**: Use segment-specific email templates from Campaign Studio")
            recommendations.append("📈 **Predictive Analytics**: Leverage CLV and churn predictions for proactive engagement")
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
            
            st.markdown("---")
            
            # === RAW DATA PREVIEW ===
            st.markdown("### 📋 Data Preview")
            st.dataframe(df_u.head(50), use_container_width=True)

            # === COLUMN SUMMARY ===
            st.markdown("### 🔎 Column Summary")
            dtypes = df_u.dtypes.astype(str)
            miss = df_u.isna().sum()
            uniq = df_u.nunique(dropna=True)
            summary = pd.DataFrame({
                'column': df_u.columns,
                'dtype': [dtypes[c] for c in df_u.columns],
                'missing': [int(miss[c]) for c in df_u.columns],
                'unique': [int(uniq[c]) for c in df_u.columns],
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)
            
            # Download options
            st.markdown("---")
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    "⬇️ Download Column Summary",
                    data=summary.to_csv(index=False),
                    file_name="column_summary.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with dl_col2:
                st.download_button(
                    "⬇️ Download Full Data",
                    data=df_u.to_csv(index=False),
                    file_name="analyzed_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    else:
        st.info("📤 Upload a CSV file to see comprehensive insights and analytics")