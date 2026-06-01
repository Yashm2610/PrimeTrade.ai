# PrimeTrade.ai: Market Sentiment vs. Trader Performance Analysis

This repository contains the data science assignment analyzing the relationship between Bitcoin market sentiment (Fear & Greed Index) and historical trader performance from the Hyperliquid perpetual DEX. 

The analysis is based on a 2-year overlapping dataset (May 1, 2023 – May 1, 2025) comprising **211,224 trade executions** across **246 assets** (with asset-specific focus on BTC).

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/Yashm2610/PrimeTrade.ai.git
cd PrimeTrade.ai
```

### 2. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install pandas numpy matplotlib seaborn scipy streamlit
```

### 3. Download the Datasets
Run the downloader script to fetch the Hyperliquid trader history and Bitcoin Fear & Greed dataset from Google Drive:
```bash
python download_data.py
```

### 4. Run the Statistical Analysis
Run the core analysis script. This script cleans and aligns the timestamps (using `Timestamp IST` to fix the CSV scientific notation precision loss), computes Pearson correlations, runs One-way ANOVA tests, saves statistics to `analysis_output.txt`, and generates static plots in the `charts/` directory:
```bash
python analysis.py
```

### 5. Launch the Interactive Web Dashboard
Run the Streamlit application to open the interactive, glassmorphic analytics dashboard in your browser:
```bash
streamlit run app.py
```
*The dashboard allows filtering by target asset (Overall vs. BTC-only), adjusting the Fear & Greed range on the fly, rendering real-time KPI metrics, showing dynamic ANOVA/correlation tests, and exploring/downloading the data.*

---

## 📊 Key Trading Insights

1. **Profitability is Counter-Cyclical**: Daily trader profits (Closed PnL) are **negatively correlated** with sentiment. Traders make the most during **Extreme Fear** ($52.8k/day average) and the least during **Greed** ($11.1k/day average).
2. **Volatility Spikes Trading Activity**: Daily trade count and volume are **strongly negatively correlated** with sentiment ($p < 0.001$). Trading counts during Extreme Fear are **4.4x higher** than during Greed.
3. **Traders are Contrarians**: Hyperliquid traders actively buy the dip in Fear (Long Open Ratio rises to **66.7%** overall, and **78.2%** for BTC-only) and short/hedge in Greed (Long Open Ratio falls to **49.2%** in Extreme Greed).
4. **Win Rate vs. PnL Asymmetry**: In Extreme Fear, traders show their **lowest win rate (65.4%)** but **highest daily PnL ($52.8k)**. This implies that they take multiple small losses attempting to catch bottoms, but their winning trades yield massive, outsized returns.

---

## 📂 Repository Structure

* `download_data.py`: Programmatic downloader script for Google Drive hosted datasets.
* `analysis.py`: Main processing script that runs the statistical tests and generates static charts.
* `app.py`: Streamlit-based interactive web dashboard with custom dark-theme glassmorphism.
* `report.md`: Deep-dive executive analysis report detailing findings, tables, and recommendations.
* `analysis_output.txt`: Captured numeric logs and statistical test outputs of the run.
* `charts/`: Folder containing saved visualization figures.
* `bitcoin_market_sentiment.csv`: The processed Bitcoin Fear & Greed index historical data.

* Added dynamic charts and statistical testing features
