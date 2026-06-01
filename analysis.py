import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, f_oneway

# Create charts directory
os.makedirs("charts", exist_ok=True)

# Set style for seaborn
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 16
})

# 1. Load Data
print("Loading datasets...")
sentiment_df = pd.read_csv("bitcoin_market_sentiment.csv")
trader_df = pd.read_csv("historical_trader_data.csv")

# 2. Date Cleaning & Alignment
print("Cleaning dates...")
sentiment_df['Date'] = pd.to_datetime(sentiment_df['date']).dt.date
sentiment_df = sentiment_df.rename(columns={'value': 'sentiment_score', 'classification': 'sentiment_class'})

trader_df['parsed_date'] = pd.to_datetime(trader_df['Timestamp IST'], format='%d-%m-%Y %H:%M')
trader_df['Date'] = trader_df['parsed_date'].dt.date

# 3. Separate Overall and BTC-only
trader_btc = trader_df[trader_df['Coin'] == 'BTC'].copy()

# 4. Define Aggregation Function
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
                                 np.nan)
    
    total_opens = daily['open_long_count'] + daily['open_short_count']
    daily['long_open_ratio'] = np.where(total_opens > 0, 
                                        daily['open_long_count'] / total_opens, 
                                        np.nan)
                                        
    total_sides = daily['buy_side_count'] + daily['sell_side_count']
    daily['buy_ratio'] = np.where(total_sides > 0,
                                  daily['buy_side_count'] / total_sides,
                                  np.nan)
    return daily

print("Aggregating daily metrics...")
daily_overall = compute_daily_metrics(trader_df)
daily_btc = compute_daily_metrics(trader_btc)

# Merge with sentiment data
merged_overall = pd.merge(daily_overall, sentiment_df[['Date', 'sentiment_score', 'sentiment_class']], on='Date', how='inner')
merged_btc = pd.merge(daily_btc, sentiment_df[['Date', 'sentiment_score', 'sentiment_class']], on='Date', how='inner')

# Open output log file
log_file = open("analysis_output.txt", "w")

def log_print(msg=""):
    print(msg)
    log_file.write(msg + "\n")

log_print("="*60)
log_print("  BITCOIN MARKET SENTIMENT VS TRADER PERFORMANCE ANALYSIS")
log_print("="*60)
log_print(f"Analysis Period: {merged_overall['Date'].min()} to {merged_overall['Date'].max()}")
log_print(f"Total days analyzed (Overall): {len(merged_overall)}")
log_print(f"Total days analyzed (BTC-Only): {len(merged_btc)}")
log_print("\n")

# 5. Correlation Analysis
cols_to_corr = ['total_pnl', 'mean_pnl', 'median_pnl', 'total_volume', 'mean_volume', 'trade_count', 'win_rate', 'long_open_ratio', 'buy_ratio', 'total_fee']

log_print("1. PEARSON CORRELATION ANALYSIS (VS SENTIMENT SCORE)")
log_print("-"*60)
log_print(f"{'Metric':<20} | {'Overall Corr':<12} | {'Overall p-val':<13} | {'BTC-Only Corr':<12} | {'BTC p-val':<10}")
log_print("-"*60)

for col in cols_to_corr:
    # Overall
    df_o = merged_overall[[col, 'sentiment_score']].dropna()
    corr_o, p_o = pearsonr(df_o[col], df_o['sentiment_score'])
    
    # BTC
    df_b = merged_btc[[col, 'sentiment_score']].dropna()
    if len(df_b) > 2:
        corr_b, p_b = pearsonr(df_b[col], df_b['sentiment_score'])
    else:
        corr_b, p_b = np.nan, np.nan
        
    log_print(f"{col:<20} | {corr_o:>12.4f} | {p_o:>13.4e} | {corr_b:>12.4f} | {p_b:>10.4e}")

log_print("\n")

# 6. Categorical Analysis Grouped by Sentiment Class
sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']

log_print("2. MEAN PERFORMANCE BY SENTIMENT CLASSIFICATION")
log_print("-"*60)

for label, df in [("OVERALL HYPERLIQUID TRADING", merged_overall), ("BTC-ONLY TRADING", merged_btc)]:
    log_print(f"\n--- {label} ---")
    grouped = df.groupby('sentiment_class')[['total_pnl', 'win_rate', 'total_volume', 'trade_count', 'long_open_ratio', 'total_fee']].mean().reindex(sentiment_order)
    log_print(grouped.to_string())
    
    # Statistical significance testing via ANOVA
    log_print("\nANOVA Test p-values (Testing difference across classes):")
    for col in ['total_pnl', 'win_rate', 'total_volume', 'trade_count', 'long_open_ratio']:
        groups = [df[df['sentiment_class'] == c][col].dropna() for c in sentiment_order]
        # Filter out empty groups
        groups = [g for g in groups if len(g) > 0]
        if len(groups) > 1:
            stat, pval = f_oneway(*groups)
            log_print(f"  {col:<20} : F-stat = {stat:.4f}, p-value = {pval:.4e}")
        else:
            log_print(f"  {col:<20} : Not enough data groups")

log_print("\n")

# 7. Generate Visualizations

# Colors matching the sentiment categories
# Fear to Greed: red/orange -> neutral grey/yellow -> light green/dark green
sentiment_colors = {
    'Extreme Fear': '#d73027',  # Red
    'Fear': '#fc8d59',          # Orange
    'Neutral': '#fee090',       # Yellow
    'Greed': '#91bfdb',         # Light Blue/Green
    'Extreme Greed': '#4575b4'  # Dark Blue/Green
}
palette = [sentiment_colors[c] for c in sentiment_order]

# CHART 1: Daily Total PnL by Sentiment Category (Overall vs BTC-Only)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.barplot(
    data=merged_overall, x='sentiment_class', y='total_pnl', 
    order=sentiment_order, palette=palette, ax=axes[0], errorbar='se', hue='sentiment_class', legend=False
)
axes[0].set_title("Overall Daily Total PnL vs. Market Sentiment")
axes[0].set_xlabel("Bitcoin Fear & Greed Classification")
axes[0].set_ylabel("Average Daily Total PnL (USD)")
axes[0].tick_params(axis='x', rotation=15)

sns.barplot(
    data=merged_btc, x='sentiment_class', y='total_pnl', 
    order=sentiment_order, palette=palette, ax=axes[1], errorbar='se', hue='sentiment_class', legend=False
)
axes[1].set_title("BTC-Only Daily Total PnL vs. Market Sentiment")
axes[1].set_xlabel("Bitcoin Fear & Greed Classification")
axes[1].set_ylabel("Average Daily Total PnL (USD)")
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig("charts/daily_pnl_by_sentiment.png", dpi=150)
plt.close()
log_print("Generated Chart 1: daily_pnl_by_sentiment.png")

# CHART 2: Trading Activity (Volume & Count) by Sentiment Category (Overall)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.barplot(
    data=merged_overall, x='sentiment_class', y='trade_count', 
    order=sentiment_order, palette=palette, ax=axes[0], errorbar='se', hue='sentiment_class', legend=False
)
axes[0].set_title("Average Daily Trade Count vs. Market Sentiment")
axes[0].set_xlabel("Bitcoin Fear & Greed Classification")
axes[0].set_ylabel("Average Daily Trade Count")
axes[0].tick_params(axis='x', rotation=15)

sns.barplot(
    data=merged_overall, x='sentiment_class', y='total_volume', 
    order=sentiment_order, palette=palette, ax=axes[1], errorbar='se', hue='sentiment_class', legend=False
)
axes[1].set_title("Average Daily Trading Volume (USD) vs. Market Sentiment")
axes[1].set_xlabel("Bitcoin Fear & Greed Classification")
axes[1].set_ylabel("Average Daily Volume (USD)")
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig("charts/trading_activity_by_sentiment.png", dpi=150)
plt.close()
log_print("Generated Chart 2: trading_activity_by_sentiment.png")

# CHART 3: Long Opening Ratio vs. Market Sentiment (Overall vs BTC-Only)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.barplot(
    data=merged_overall, x='sentiment_class', y='long_open_ratio', 
    order=sentiment_order, palette=palette, ax=axes[0], errorbar='se', hue='sentiment_class', legend=False
)
axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.7)
axes[0].set_title("Overall Long Open Ratio vs. Market Sentiment")
axes[0].set_xlabel("Bitcoin Fear & Greed Classification")
axes[0].set_ylabel("Long Open Ratio (Open Long / Total Open Trades)")
axes[0].set_ylim(0.4, 0.9)
axes[0].tick_params(axis='x', rotation=15)

sns.barplot(
    data=merged_btc, x='sentiment_class', y='long_open_ratio', 
    order=sentiment_order, palette=palette, ax=axes[1], errorbar='se', hue='sentiment_class', legend=False
)
axes[1].axhline(0.5, color='gray', linestyle='--', alpha=0.7)
axes[1].set_title("BTC-Only Long Open Ratio vs. Market Sentiment")
axes[1].set_xlabel("Bitcoin Fear & Greed Classification")
axes[1].set_ylabel("Long Open Ratio (Open Long / Total Open Trades)")
axes[1].set_ylim(0.4, 0.9)
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig("charts/long_open_ratio_by_sentiment.png", dpi=150)
plt.close()
log_print("Generated Chart 3: long_open_ratio_by_sentiment.png")

# CHART 4: Scatter plots with regression lines showing continuous trends
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Sentiment vs Long Open Ratio (Overall)
sns.regplot(
    data=merged_overall, x='sentiment_score', y='long_open_ratio',
    scatter_kws={'alpha':0.5, 'color': '#7f8c8d'}, line_kws={'color': '#e74c3c'}, ax=axes[0]
)
axes[0].set_title("Sentiment Score vs. Long Open Ratio")
axes[0].set_xlabel("Bitcoin Fear & Greed Score")
axes[0].set_ylabel("Long Open Ratio")

# Plot 2: Sentiment vs Trade Count (Log Scale) (Overall)
merged_overall['log_trade_count'] = np.log10(merged_overall['trade_count'])
sns.regplot(
    data=merged_overall, x='sentiment_score', y='log_trade_count',
    scatter_kws={'alpha':0.5, 'color': '#7f8c8d'}, line_kws={'color': '#2980b9'}, ax=axes[1]
)
axes[1].set_title("Sentiment Score vs. Trade Count (Log10)")
axes[1].set_xlabel("Bitcoin Fear & Greed Score")
axes[1].set_ylabel("Log10(Daily Trade Count)")

# Plot 3: Sentiment vs Trading Volume (Log Scale) (Overall)
merged_overall['log_total_volume'] = np.log10(merged_overall['total_volume'])
sns.regplot(
    data=merged_overall, x='sentiment_score', y='log_total_volume',
    scatter_kws={'alpha':0.5, 'color': '#7f8c8d'}, line_kws={'color': '#27ae60'}, ax=axes[2]
)
axes[2].set_title("Sentiment Score vs. Volume USD (Log10)")
axes[2].set_xlabel("Bitcoin Fear & Greed Score")
axes[2].set_ylabel("Log10(Daily Trading Volume)")

plt.tight_layout()
plt.savefig("charts/sentiment_trends.png", dpi=150)
plt.close()
log_print("Generated Chart 4: sentiment_trends.png")

log_file.close()
print("Analysis complete. Check analysis_output.txt for results.")
