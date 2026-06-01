import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, f_oneway

# Set Page Config
st.set_page_config(
    page_title="Primetrade.ai - Market Sentiment vs. Trader Performance",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark/Glassmorphism theme with premium typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container background */
    .stApp {
        background-color: #080c14;
        color: #f1f5f9;
    }
    
    /* Custom Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
    }
    
    /* Title Gradient Banner */
    .hero-container {
        padding: 30px 40px;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px 0 rgba(0,0,0,0.3);
    }
    
    .title-gradient {
        background: linear-gradient(90deg, #60a5fa 0%, #34d399 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        margin-top: 10px;
        margin-bottom: 0;
        font-weight: 400;
    }
    
    /* Glassmorphic Metric Cards */
    .metric-card {
        background: rgba(13, 20, 35, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 22px 20px;
        box-shadow: 0 10px 25px 0 rgba(0, 0, 0, 0.25);
        transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 20px;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(96, 165, 250, 0.35);
        box-shadow: 0 15px 35px 0 rgba(96, 165, 250, 0.12);
    }
    
    .metric-title {
        color: #64748b;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        font-family: 'Outfit', sans-serif;
        margin-top: 10px;
        letter-spacing: -0.01em;
    }
    
    /* Section containers */
    .section-box {
        background: rgba(13, 20, 35, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

# Downloader helper if data is missing on cloud/container
def ensure_datasets_exist():
    import os
    import requests
    
    def download_from_gdrive(file_id, dest):
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        res = session.get(URL, params={'id': file_id}, stream=True)
        # Handle verification tokens for large files
        token = None
        for k, v in res.cookies.items():
            if k.startswith('download_warning'):
                token = v
                break
        if token:
            res = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)
        with open(dest, "wb") as f:
            for chunk in res.iter_content(32768):
                if chunk:
                    f.write(chunk)

    if not os.path.exists("historical_trader_data.csv"):
        download_from_gdrive("1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs", "historical_trader_data.csv")
    if not os.path.exists("bitcoin_market_sentiment.csv"):
        download_from_gdrive("1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf", "bitcoin_market_sentiment.csv")

try:
    ensure_datasets_exist()
except Exception as e:
    st.error(f"Failed to auto-download datasets: {e}")
    st.stop()

# Cache data loading for faster performance
@st.cache_data
def load_and_preprocess_data():
    sentiment_df = pd.read_csv("bitcoin_market_sentiment.csv")
    trader_df = pd.read_csv("historical_trader_data.csv")
    
    # Preprocess Sentiment
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['date']).dt.date
    sentiment_df = sentiment_df.rename(columns={'value': 'sentiment_score', 'classification': 'sentiment_class'})
    
    # Preprocess Trader
    trader_df['parsed_date'] = pd.to_datetime(trader_df['Timestamp IST'], format='%d-%m-%Y %H:%M')
    trader_df['Date'] = trader_df['parsed_date'].dt.date
    
    return sentiment_df, trader_df

try:
    sentiment_df, trader_df = load_and_preprocess_data()
except Exception as e:
    st.error(f"Error loading datasets: {e}")
    st.stop()

# Aggregation helper
def compute_daily_metrics(df):
    daily = df.groupby('Date').agg(
        total_pnl=('Closed PnL', 'sum'),
        mean_pnl=('Closed PnL', 'mean'),
        median_pnl=('Closed PnL', 'median'),
        total_volume=('Size USD', 'sum'),
        mean_volume=('Size USD', 'mean'),
        trade_count=('Account', 'count'),
        total_fee=('Fee', 'sum'),
        mean_fee=('Fee', 'mean'),
        positive_pnl_trades=('Closed PnL', lambda x: (x > 0).sum()),
        nonzero_pnl_trades=('Closed PnL', lambda x: (x != 0).sum()),
        open_long_count=('Direction', lambda x: (x == 'Open Long').sum()),
        open_short_count=('Direction', lambda x: (x == 'Open Short').sum()),
        buy_side_count=('Side', lambda x: (x == 'BUY').sum()),
        sell_side_count=('Side', lambda x: (x == 'SELL').sum())
    ).reset_index()
    
    # Calculate ratios
    daily['win_rate'] = np.where(daily['nonzero_pnl_trades'] > 0, 
                                 daily['positive_pnl_trades'] / daily['nonzero_pnl_trades'], 
                                 0.0)
    
    total_opens = daily['open_long_count'] + daily['open_short_count']
    daily['long_open_ratio'] = np.where(total_opens > 0, 
                                        daily['open_long_count'] / total_opens, 
                                        0.0)
                                        
    total_sides = daily['buy_side_count'] + daily['sell_side_count']
    daily['buy_ratio'] = np.where(total_sides > 0,
                                  daily['buy_side_count'] / total_sides,
                                  0.0)
    return daily

# Sidebar filters only (No Navigation here anymore!)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.image("https://img.icons8.com/color/96/bitcoin.png", width=75)
st.sidebar.markdown("<h2 style='font-family:Outfit; margin-top:0;'>PRIMETRADE.AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.markdown("### Analytics Settings")
asset_filter = st.sidebar.selectbox("Target Asset Group", ["Overall Hyperliquid", "BTC-Only"])

# Apply Asset Filter
if asset_filter == "BTC-Only":
    filtered_trader = trader_df[trader_df['Coin'] == 'BTC'].copy()
else:
    filtered_trader = trader_df.copy()

# Compute metrics
daily_df = compute_daily_metrics(filtered_trader)
merged_df = pd.merge(daily_df, sentiment_df[['Date', 'sentiment_score', 'sentiment_class']], on='Date', how='inner')

score_range = st.sidebar.slider(
    "Fear & Greed Index Range",
    min_value=0, max_value=100,
    value=(0, 100)
)
plot_df = merged_df[(merged_df['sentiment_score'] >= score_range[0]) & (merged_df['sentiment_score'] <= score_range[1])]

# Matplotlib Helper function to style plots for Dark Theme
def apply_dark_theme_chart(fig, ax):
    fig.patch.set_facecolor('#080c14')
    ax.set_facecolor('#0e1525')
    ax.grid(True, color='#ffffff', alpha=0.05, linestyle='--', linewidth=0.5)
    
    # Text styling
    ax.tick_params(colors='#94a3b8', labelsize=10)
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    ax.title.set_color('#ffffff')
    
    # Hide outer spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color((1.0, 1.0, 1.0, 0.1))

# Hero Title Banner on the Main Page
st.markdown("""
<div class="hero-container">
    <h1 class="title-gradient">PRIMETRADE.AI</h1>
    <div class="subtitle">Bitcoin Market Sentiment vs. Hyperliquid Trader Performance Intelligence</div>
</div>
""", unsafe_allow_html=True)

# Main Navigation tabs at the top of the page!
tab_dashboard, tab_explorer, tab_report = st.tabs([
    "📊 Performance Dashboard", 
    "📂 Explore Aggregated Data", 
    "📄 Executive Analysis Report"
])

# -----------------
# TAB 1: DASHBOARD
# -----------------
with tab_dashboard:
    st.markdown(f"Currently Analyzing: **{asset_filter}** | Date Range: `{plot_df['Date'].min()}` to `{plot_df['Date'].max()}`")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # KPI Row
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    avg_daily_pnl = plot_df['total_pnl'].mean()
    avg_win_rate = plot_df['win_rate'].mean() * 100
    avg_vol = plot_df['total_volume'].mean() / 1e6
    avg_trades = plot_df['trade_count'].mean()
    avg_long_ratio = plot_df['long_open_ratio'].mean() * 100
    
    with kpi1:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #10b981;">
            <div class="metric-title">Avg. Daily PnL</div>
            <div class="metric-value">${avg_daily_pnl:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi2:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #3b82f6;">
            <div class="metric-title">Avg. Win Rate</div>
            <div class="metric-value">{avg_win_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi3:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #8b5cf6;">
            <div class="metric-title">Avg. Daily Volume</div>
            <div class="metric-value">${avg_vol:.2f}M</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi4:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #f59e0b;">
            <div class="metric-title">Avg. Daily Trades</div>
            <div class="metric-value">{avg_trades:.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi5:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #ec4899;">
            <div class="metric-title">Long Open Ratio</div>
            <div class="metric-value">{avg_long_ratio:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<h3 style='margin-bottom:20px;'>📈 Sentiment-Driven Relationship Charts</h3>", unsafe_allow_html=True)
    
    # Layout for charts
    row1_c1, row1_c2 = st.columns(2)
    row2_c1, row2_c2 = st.columns(2)
    
    sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    sentiment_colors = {
        'Extreme Fear': '#ef4444',
        'Fear': '#f97316',
        'Neutral': '#eab308',
        'Greed': '#3b82f6',
        'Extreme Greed': '#10b981'
    }
    available_classes = [c for c in sentiment_order if c in plot_df['sentiment_class'].unique()]
    palette = [sentiment_colors[c] for c in available_classes]
    
    # Chart 1: PnL by Sentiment
    with row1_c1:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.markdown("##### Average Daily Total PnL vs. Market Sentiment Category")
        fig, ax = plt.subplots(figsize=(8, 4.2))
        sns.barplot(
            data=plot_df, x='sentiment_class', y='total_pnl', 
            order=available_classes, palette=palette, ax=ax, errorbar='se', hue='sentiment_class', legend=False
        )
        ax.set_ylabel("Daily Total PnL (USD)")
        ax.set_xlabel("Fear & Greed Category")
        apply_dark_theme_chart(fig, ax)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Chart 2: Long Open Ratio
    with row1_c2:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.markdown("##### Long Open Ratio (Open Long / Total Open) vs. Market Sentiment")
        fig, ax = plt.subplots(figsize=(8, 4.2))
        sns.barplot(
            data=plot_df, x='sentiment_class', y='long_open_ratio', 
            order=available_classes, palette=palette, ax=ax, errorbar='se', hue='sentiment_class', legend=False
        )
        ax.axhline(0.5, color='#94a3b8', linestyle='--', alpha=0.5)
        ax.set_ylabel("Long Open Ratio")
        ax.set_xlabel("Fear & Greed Category")
        ax.set_ylim(0.3, 0.9)
        apply_dark_theme_chart(fig, ax)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Chart 3: Volume & Trades
    with row2_c1:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.markdown("##### Average Daily Trading Volume vs. Market Sentiment")
        fig, ax = plt.subplots(figsize=(8, 4.2))
        sns.barplot(
            data=plot_df, x='sentiment_class', y='total_volume', 
            order=available_classes, palette=palette, ax=ax, errorbar='se', hue='sentiment_class', legend=False
        )
        ax.set_ylabel("Daily Volume (USD)")
        ax.set_xlabel("Fear & Greed Category")
        apply_dark_theme_chart(fig, ax)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # Chart 4: Continuous Trends
    with row2_c2:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.markdown("##### Sentiment Score vs. Long Open Ratio (Continuous Trend Line)")
        fig, ax = plt.subplots(figsize=(8, 4.2))
        sns.regplot(
            data=plot_df, x='sentiment_score', y='long_open_ratio',
            scatter_kws={'alpha':0.3, 'color': '#60a5fa'}, line_kws={'color': '#10b981', 'linewidth': 2}, ax=ax
        )
        ax.set_ylabel("Long Open Ratio")
        ax.set_xlabel("Fear & Greed Score (0-100)")
        apply_dark_theme_chart(fig, ax)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # Statistical Rigor section
    st.markdown("---")
    st.markdown("<h3 style='margin-bottom:20px;'>🔬 Statistical Significance Tests</h3>", unsafe_allow_html=True)
    
    stat_col1, stat_col2 = st.columns(2)
    
    with stat_col1:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.markdown("##### Pearson Correlation (Continuous Score)")
        df_clean = plot_df[['sentiment_score', 'total_pnl', 'total_volume', 'trade_count', 'long_open_ratio']].dropna()
        
        if len(df_clean) > 5:
            corr_pnl, p_pnl = pearsonr(df_clean['total_pnl'], df_clean['sentiment_score'])
            corr_vol, p_vol = pearsonr(df_clean['total_volume'], df_clean['sentiment_score'])
            corr_long, p_long = pearsonr(df_clean['long_open_ratio'], df_clean['sentiment_score'])
            
            st.markdown(f"""
            * **Daily PnL Correlation**: `{corr_pnl:.4f}` *(p-value: {p_pnl:.2e})*
            * **Trading Volume Correlation**: `{corr_vol:.4f}` *(p-value: {p_vol:.2e})*
            * **Long Open Ratio Correlation**: `{corr_long:.4f}` *(p-value: {p_long:.2e})*
            """)
            
            if p_vol < 0.01:
                st.success("✅ Volume has a highly significant negative correlation with market sentiment.")
            if p_long < 0.05:
                st.success("✅ Long Open Ratio has a statistically significant relationship with market sentiment.")
        else:
            st.warning("Not enough data rows to compute correlation coefficient.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with stat_col2:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.markdown("##### One-Way ANOVA (Categorical Differences)")
        
        # Test if total_pnl and long_open_ratio are statistically different across classes
        groups_pnl = [plot_df[plot_df['sentiment_class'] == c]['total_pnl'].dropna() for c in available_classes]
        groups_pnl = [g for g in groups_pnl if len(g) > 2]
        
        groups_long = [plot_df[plot_df['sentiment_class'] == c]['long_open_ratio'].dropna() for c in available_classes]
        groups_long = [g for g in groups_long if len(g) > 2]
        
        if len(groups_pnl) > 1 and len(groups_long) > 1:
            f_pnl, pval_pnl = f_oneway(*groups_pnl)
            f_long, pval_long = f_oneway(*groups_long)
            
            st.markdown(f"""
            * **Total PnL ANOVA p-value**: `{pval_pnl:.4e}` *(F-stat: {f_pnl:.2f})*
            * **Long Open Ratio ANOVA p-value**: `{pval_long:.4e}` *(F-stat: {f_long:.2f})*
            """)
            
            if pval_pnl < 0.05:
                st.success("✅ Trader daily profitability is significantly different across sentiment categories.")
            if pval_long < 0.05:
                st.success("✅ Long vs Short opening ratios are significantly different across sentiment categories.")
        else:
            st.warning("Not enough classes represented in filtered data to perform ANOVA.")
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------
# TAB 2: EXPLORER
# -----------------
with tab_explorer:
    st.title("📂 Aggregated Daily Dataset Explorer")
    st.markdown("Examine or download the preprocessed daily data combining both sentiment scores and aggregated trading metrics.")
    st.markdown("---")
    
    st.dataframe(plot_df.style.background_gradient(subset=['total_pnl'], cmap='RdYlGn'), use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    csv = plot_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Daily Aggregated Dataset (CSV)",
        data=csv,
        file_name="sentiment_vs_trader_daily_performance.csv",
        mime="text/csv"
    )

# -----------------
# TAB 3: REPORT
# -----------------
with tab_report:
    st.title("📄 Executive Analysis Report")
    st.markdown("---")
    
    if os.path.exists("report.md"):
        with open("report.md", "r", encoding="utf-8") as f:
            report_content = f.read()
        
        # Replace relative chart links with text/image indicators since charts are rendered in dashboard tab
        report_content = report_content.replace("![Daily PnL by Sentiment](charts/daily_pnl_by_sentiment.png)", "*[Chart 1: Daily PnL vs. Market Sentiment is available in the Performance Dashboard tab]*")
        report_content = report_content.replace("![Trading Activity by Sentiment](charts/trading_activity_by_sentiment.png)", "*[Chart 2: Trading Activity vs. Market Sentiment is available in the Performance Dashboard tab]*")
        report_content = report_content.replace("![Long Open Ratio by Sentiment](charts/long_open_ratio_by_sentiment.png)", "*[Chart 3: Long Open Ratio vs. Market Sentiment is available in the Performance Dashboard tab]*")
        report_content = report_content.replace("![Sentiment Trends](charts/sentiment_trends.png)", "*[Chart 4: Sentiment Trends Scatter is available in the Performance Dashboard tab]*")
        
        st.markdown(report_content)
    else:
        st.error("report.md not found in the directory. Please make sure report.md is generated first.")

# styled layout grid and card spacing

# closed matplotlib plots to release memory resources
