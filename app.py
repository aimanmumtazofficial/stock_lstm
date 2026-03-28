# import math
# import datetime

# import warnings

# import streamlit as st
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
# import tensorflow as tf
# from tensorflow.keras import layers, models, callbacks

# warnings.filterwarnings("ignore")
# tf.get_logger().setLevel("ERROR")

# # ── yfinance import — show clear fix if still on old version ──────
# try:
#     import yfinance as yf
#     _yf_ver = tuple(int(x) for x in yf.__version__.split(".")[:3])
#     if _yf_ver < (0, 2, 40):
#         st.warning(
#             f"⚠️ You have yfinance **{yf.__version__}** — this version is too old and "
#             "will fail with 'JSONDecodeError' or 'No timezone found'.\n\n"
#             "Run this in your terminal, then restart Streamlit:\n"
#             "```\npip uninstall yfinance multitasking -y\n"
#             'pip install "yfinance>=0.2.40"\n```'
#         )
# except TypeError:
#     st.error(
#         "yfinance import failed (Python 3.8 + old multitasking conflict).\n\n"
#         "Fix:\n```\npip uninstall yfinance multitasking -y\n"
#         'pip install "yfinance>=0.2.40"\n```'
#     )
#     st.stop()
# except Exception as e:
#     st.error(f"Cannot import yfinance: {e}")
#     st.stop()

# # ══════════════════════════════════════════════════════════════════
# # PAGE CONFIG
# # ══════════════════════════════════════════════════════════════════
# st.set_page_config(
#     page_title="LSTM Stock Analyzer",
#     page_icon="📈",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# st.markdown("""
# <style>
#   [data-testid="stAppViewContainer"] { background: #050a0f; }
#   [data-testid="stSidebar"]          { background: #080f18; border-right:1px solid #1a3050; }
#   .stButton > button {
#     background:#00e5ff15; border:1px solid #00e5ff;
#     color:#00e5ff; font-family:monospace; letter-spacing:.06em; transition:all .2s;
#   }
#   .stButton > button:hover { background:#00e5ff30; }
#   h1,h2,h3 { color:#c8dff0 !important; }
#   .stProgress > div > div { background:linear-gradient(90deg,#00e5ff,#00ff9d) !important; }
#   code { color:#00e5ff !important; background:#0a1520 !important; }
#   [data-testid="stMarkdownContainer"] p { color:#c8dff0; }
#   div[data-testid="metric-container"] { background:#0a1520; border:1px solid #0f2035;
#                                         padding:.75rem; border-radius:4px; }
# </style>
# """, unsafe_allow_html=True)

# # ══════════════════════════════════════════════════════════════════
# # TICKER CATALOGUE — dropdown instead of free-text input
# # ══════════════════════════════════════════════════════════════════
# TICKERS = {
#     # ── US Index ETFs ──────────────────────────────────────────────
#     "SPY  — S&P 500 ETF (top 500 US companies)":               "SPY",
#     "QQQ  — NASDAQ 100 ETF (tech-heavy)":                      "QQQ",
#     "DIA  — Dow Jones Industrial Average ETF":                 "DIA",
#     "IWM  — Russell 2000 Small-Cap ETF":                       "IWM",
#     "VTI  — Total US Stock Market ETF":                        "VTI",
#     "VOO  — Vanguard S&P 500 ETF":                             "VOO",
#     # ── Sector ETFs ────────────────────────────────────────────────
#     "XLK  — Technology Sector ETF":                            "XLK",
#     "XLF  — Financial Sector ETF":                             "XLF",
#     "XLE  — Energy Sector ETF":                                "XLE",
#     "XLV  — Healthcare Sector ETF":                            "XLV",
#     "XLY  — Consumer Discretionary ETF":                       "XLY",
#     # ── Big Tech Stocks ────────────────────────────────────────────
#     "AAPL — Apple Inc.":                                       "AAPL",
#     "MSFT — Microsoft Corporation":                            "MSFT",
#     "GOOGL— Alphabet (Google)":                                "GOOGL",
#     "AMZN — Amazon.com Inc.":                                  "AMZN",
#     "NVDA — NVIDIA Corporation":                               "NVDA",
#     "META — Meta Platforms (Facebook)":                        "META",
#     "TSLA — Tesla Inc. (high volatility)":                     "TSLA",
#     # ── Other Assets ───────────────────────────────────────────────
#     "GLD  — Gold ETF":                                         "GLD",
#     "TLT  — US Treasury 20yr Bond ETF":                        "TLT",
#     "BTC-USD — Bitcoin (USD)":                                 "BTC-USD",
# }

# # ══════════════════════════════════════════════════════════════════
# # SIDEBAR
# # ══════════════════════════════════════════════════════════════════
# with st.sidebar:
#     st.markdown("### ⟨ LSTM Fund Analyzer ⟩")
#     st.markdown("---")

#     # Dropdown instead of text input
#     ticker_label = st.selectbox(
#         "Select Asset",
#         options=list(TICKERS.keys()),
#         index=0,
#         help="Choose a stock or ETF to analyze"
#     )
#     ticker = TICKERS[ticker_label]
#     st.caption(f"Ticker symbol: **`{ticker}`**")

#     period_label = st.selectbox(
#         "Historical Data Period",
#         options=["1 Year", "2 Years", "5 Years", "10 Years", "Max Available"],
#         index=2,
#         help="More data = better LSTM training. Recommended: 5 Years minimum."
#     )
#     period_map = {
#         "1 Year": 1, "2 Years": 2, "5 Years": 5,
#         "10 Years": 10, "Max Available": 20
#     }
#     period_years = period_map[period_label]

#     window_size = st.slider(
#         "Lookback Window (trading days)",
#         min_value=20, max_value=120, value=60, step=5,
#         help="How many past days the LSTM sees before predicting"
#     )
#     forecast_days = st.slider(
#         "Forecast Horizon (days ahead)",
#         min_value=5, max_value=30, value=10, step=5,
#         help="How many days into the future the model predicts"
#     )
#     return_threshold = st.slider(
#         "BUY Signal Threshold (%)",
#         min_value=0.5, max_value=5.0, value=1.5, step=0.5,
#         help="Min expected % gain to classify a day as BUY"
#     ) / 100.0

#     st.markdown("---")
#     st.markdown("**Model Architecture**")
#     lstm_units_1 = st.select_slider("LSTM Layer 1 Units", [32, 64, 128, 256], value=128)
#     lstm_units_2 = st.select_slider("LSTM Layer 2 Units", [16, 32, 64, 128], value=64)
#     dropout_rate = st.slider("Dropout Rate", 0.1, 0.5, 0.3, 0.05)
#     epochs       = st.slider("Max Training Epochs", 20, 150, 80, 10)

#     st.markdown("---")
#     run_btn = st.button("🚀  Run Analysis", use_container_width=True)


# # ══════════════════════════════════════════════════════════════════
# # DATA LOADING — robust, handles yfinance API changes
# # ══════════════════════════════════════════════════════════════════

# @st.cache_resource(show_spinner=False)
# def load_data(ticker: str, years: int) -> pd.DataFrame:
#     """
#     Download OHLCV data using explicit start/end dates.

#     Uses Ticker().history() — the most stable method in yfinance >= 0.2.40.
#     Strips timezone from the DatetimeIndex so downstream pandas operations
#     (comparisons, merges) work without tz-aware vs tz-naive errors.
#     """
#     end_dt   = datetime.date.today()
#     start_dt = end_dt - datetime.timedelta(days=365 * years)

#     tk = yf.Ticker(ticker)
#     df = tk.history(
#         start=start_dt.strftime("%Y-%m-%d"),
#         end=end_dt.strftime("%Y-%m-%d"),
#         interval="1d",
#         auto_adjust=True,   # adjusts for splits and dividends automatically
#         actions=False       # we don't need dividend/split columns
#     )

#     # Flatten MultiIndex columns if present (some yfinance versions return them)
#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = df.columns.get_level_values(0)

#     # Strip timezone — yfinance returns tz-aware index, pandas operations
#     # can fail when mixing tz-aware and tz-naive timestamps later
#     if hasattr(df.index, "tz") and df.index.tz is not None:
#         df.index = df.index.tz_localize(None)

#     # Keep only the OHLCV columns we need
#     needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
#     df = df[needed].dropna()
#     return df


# # ══════════════════════════════════════════════════════════════════
# # FEATURE ENGINEERING
# # ══════════════════════════════════════════════════════════════════

# def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Build 17 candlestick + technical indicator features.
#     No look-ahead bias: every value only uses data available on or before that day.
#     """
#     f = pd.DataFrame(index=df.index)

#     # ── Candlestick body features ──────────────────────────────────
#     f["body"]         = df["Close"] - df["Open"]
#     f["body_ratio"]   = f["body"].abs() / (df["High"] - df["Low"] + 1e-8)
#     f["upper_shadow"] = df["High"] - df[["Open", "Close"]].max(axis=1)
#     f["lower_shadow"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
#     f["direction"]    = np.sign(f["body"])

#     # ── Return features ────────────────────────────────────────────
#     f["return_1d"]  = df["Close"].pct_change(1)
#     f["return_5d"]  = df["Close"].pct_change(5)
#     f["return_10d"] = df["Close"].pct_change(10)

#     # ── Trend / moving average features ───────────────────────────
#     sma20 = df["Close"].rolling(20).mean()
#     sma50 = df["Close"].rolling(50).mean()
#     f["price_vs_sma20"] = df["Close"] / (sma20 + 1e-8)
#     f["price_vs_sma50"] = df["Close"] / (sma50 + 1e-8)
#     f["sma_cross"]      = sma20 / (sma50 + 1e-8)

#     ema12 = df["Close"].ewm(span=12, adjust=False).mean()
#     ema26 = df["Close"].ewm(span=26, adjust=False).mean()
#     f["macd"]           = (ema12 - ema26) / (df["Close"] + 1e-8)

#     # ── Volatility features ────────────────────────────────────────
#     f["volatility_20d"] = f["return_1d"].rolling(20).std()
#     f["atr_ratio"]      = (df["High"] - df["Low"]).rolling(14).mean() / (df["Close"] + 1e-8)

#     # ── Volume features ────────────────────────────────────────────
#     vol_avg20          = df["Volume"].rolling(20).mean()
#     f["volume_ratio"]  = df["Volume"] / (vol_avg20 + 1)
#     obv                = (np.sign(f["return_1d"]) * df["Volume"]).cumsum()
#     f["obv_ratio"]     = obv / (obv.abs().rolling(20).mean() + 1)

#     # ── RSI (Relative Strength Index) — normalized to [0,1] ────────
#     delta = df["Close"].diff()
#     gain  = delta.clip(lower=0).rolling(14).mean()
#     loss  = (-delta.clip(upper=0)).rolling(14).mean()
#     f["rsi"] = (100 - 100 / (1 + gain / (loss + 1e-8))) / 100.0

#     return f.dropna()


# # ══════════════════════════════════════════════════════════════════
# # LABELS, SEQUENCES, MODEL
# # ══════════════════════════════════════════════════════════════════

# def create_labels(df_orig, feat_index, horizon, threshold):
#     aligned = df_orig.loc[feat_index, "Close"]
#     ret     = aligned.shift(-horizon) / aligned - 1
#     return (ret > threshold).astype(int)


# def make_sequences(features_arr, labels_arr, window):
#     X, y = [], []
#     for i in range(window, len(features_arr)):
#         X.append(features_arr[i - window: i])
#         y.append(labels_arr[i])
#     return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# def build_lstm_model(window, n_features, units1, units2, drop):
#     """
#     2-layer stacked LSTM classifier using tf.keras built-in layers only.
#     Input  shape: (batch, window, n_features)
#     Output shape: (batch, 1) — sigmoid probability of BUY
#     """
#     mdl = models.Sequential([
#         layers.LSTM(units1, return_sequences=True,
#                     input_shape=(window, n_features), name="lstm_1"),
#         layers.Dropout(drop, name="drop_1"),
#         layers.LSTM(units2, return_sequences=False, name="lstm_2"),
#         layers.Dropout(drop, name="drop_2"),
#         layers.Dense(32, activation="relu", name="dense"),
#         layers.Dropout(drop / 2, name="drop_3"),
#         layers.Dense(1, activation="sigmoid", name="output")
#     ], name="LSTM_Fund_Analyzer")
#     mdl.compile(
#         optimizer=tf.keras.optimizers.Adam(0.0005),
#         loss="binary_crossentropy",
#         metrics=["accuracy"]
#     )
#     return mdl


# def decode_signal(prob):
#     if prob >= 0.65:   return "STRONG BUY",  "buy",  "🟢"
#     elif prob >= 0.52: return "BUY",          "buy",  "🟩"
#     elif prob >= 0.35: return "HOLD / WAIT",  "hold", "🟡"
#     elif prob >= 0.20: return "SELL",         "sell", "🟥"
#     else:              return "STRONG SELL",  "sell", "🔴"


# # ══════════════════════════════════════════════════════════════════
# # CHART HELPERS
# # ══════════════════════════════════════════════════════════════════

# def candlestick_fig(df, ticker):
#     fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
#                         row_heights=[0.75, 0.25], vertical_spacing=0.03)
#     fig.add_trace(go.Candlestick(
#         x=df.index, open=df["Open"], high=df["High"],
#         low=df["Low"], close=df["Close"],
#         increasing_line_color="#00ff9d", decreasing_line_color="#ff4d4d",
#         name=ticker), row=1, col=1)
#     colors = ["#00ff9d" if c >= o else "#ff4d4d"
#               for c, o in zip(df["Close"], df["Open"])]
#     fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
#                          marker_color=colors, opacity=0.5, name="Volume"),
#                   row=2, col=1)
#     fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(20).mean(),
#                              line=dict(color="#00e5ff", width=1.2), name="SMA 20"),
#                   row=1, col=1)
#     fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(50).mean(),
#                              line=dict(color="#b06cff", width=1.2), name="SMA 50"),
#                   row=1, col=1)
#     fig.update_layout(plot_bgcolor="#050a0f", paper_bgcolor="#050a0f",
#                       font_color="#c8dff0", height=520,
#                       xaxis_rangeslider_visible=False,
#                       legend=dict(bgcolor="#0a1520", bordercolor="#1a3050", borderwidth=1),
#                       margin=dict(l=10, r=10, t=30, b=10))
#     fig.update_xaxes(gridcolor="#0f2035"); fig.update_yaxes(gridcolor="#0f2035")
#     return fig


# def loss_fig(history):
#     ep  = list(range(1, len(history.history["loss"]) + 1))
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=ep, y=history.history["loss"],
#                              name="Train Loss", line=dict(color="#00e5ff", width=2)))
#     fig.add_trace(go.Scatter(x=ep, y=history.history["val_loss"],
#                              name="Val Loss",
#                              line=dict(color="#ffb800", width=2, dash="dash")))
#     fig.update_layout(plot_bgcolor="#050a0f", paper_bgcolor="#050a0f",
#                       font_color="#c8dff0", height=300,
#                       xaxis_title="Epoch", yaxis_title="Loss",
#                       legend=dict(bgcolor="#0a1520"),
#                       margin=dict(l=10, r=10, t=20, b=10))
#     fig.update_xaxes(gridcolor="#0f2035"); fig.update_yaxes(gridcolor="#0f2035")
#     return fig


# def signal_history_fig(dates, probs):
#     fig = go.Figure()
#     fig.add_hrect(y0=0.65, y1=1.0, fillcolor="#00ff9d", opacity=0.07,
#                   annotation_text="BUY zone", annotation_font_color="#00ff9d")
#     fig.add_hrect(y0=0.35, y1=0.65, fillcolor="#ffb800", opacity=0.05,
#                   annotation_text="HOLD zone", annotation_font_color="#ffb800")
#     fig.add_hrect(y0=0.0,  y1=0.35, fillcolor="#ff4d4d", opacity=0.07,
#                   annotation_text="SELL zone", annotation_font_color="#ff4d4d")
#     fig.add_trace(go.Scatter(x=dates, y=probs, mode="lines",
#                              name="BUY Probability",
#                              line=dict(color="#00e5ff", width=2),
#                              fill="tozeroy", fillcolor="rgba(0,229,255,0.05)"))
#     fig.add_hline(y=0.5, line_dash="dot", line_color="#ffffff", opacity=0.3)
#     fig.update_layout(plot_bgcolor="#050a0f", paper_bgcolor="#050a0f",
#                       font_color="#c8dff0", height=280,
#                       yaxis=dict(range=[0, 1], title="BUY Probability"),
#                       margin=dict(l=10, r=10, t=20, b=10))
#     fig.update_xaxes(gridcolor="#0f2035"); fig.update_yaxes(gridcolor="#0f2035")
#     return fig


# # ══════════════════════════════════════════════════════════════════
# # LANDING PAGE (before Run is clicked)
# # ══════════════════════════════════════════════════════════════════

# st.title("📈 LSTM Stock Fund Investment Analyzer")
# st.caption(
#     "Uses `tf.keras.layers.LSTM` (TensorFlow built-in) to detect candlestick patterns "
#     "and generate **BUY / HOLD / SELL** signals with confidence scores."
# )
# st.markdown("---")

# if not run_btn:
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         st.markdown("#### How it works")
#         st.markdown("""
# 1. Downloads real OHLCV data via Yahoo Finance
# 2. Engineers 17 candlestick + technical features
# 3. Trains a 2-layer stacked LSTM classifier
# 4. Outputs BUY / HOLD / SELL with confidence %
#         """)
#     with c2:
#         st.markdown("#### LSTM Architecture")
#         st.code("""LSTM(128, return_sequences=True)
# Dropout(0.3)
# LSTM(64,  return_sequences=False)
# Dropout(0.3)
# Dense(32, activation='relu')
# Dense(1,  activation='sigmoid')""", language="python")
#     with c3:
#         st.markdown("#### Required install")
#         st.code("""pip uninstall yfinance multitasking -y
# pip install "yfinance>=0.2.40"
# pip install streamlit plotly
# pip install scikit-learn tensorflow""", language="bash")

#     st.info("👈 Select an asset from the sidebar and click **Run Analysis**")
#     st.stop()


# # ══════════════════════════════════════════════════════════════════
# # PIPELINE EXECUTION
# # ══════════════════════════════════════════════════════════════════

# bar = st.progress(0, text="")

# # ── STEP 1: Download ──────────────────────────────────────────────
# bar.progress(5, text=f"📥 Downloading {ticker} ({period_label})...")
# try:
#     df = load_data(ticker, period_years)
# except Exception as e:
#     st.error(f"Download failed for **{ticker}**: {e}\n\n"
#              "Make sure you have run:\n"
#              "```\npip uninstall yfinance multitasking -y\n"
#              'pip install "yfinance>=0.2.40"\n```')
#     st.stop()

# if df is None or df.empty:
#     st.error(
#         f"**`{ticker}` returned empty data.**\n\n"
#         "Most likely cause: yfinance version is too old.\n\n"
#         "Fix (run in terminal, then restart Streamlit):\n"
#         "```\npip uninstall yfinance multitasking -y\n"
#         'pip install "yfinance>=0.2.40"\n```'
#     )
#     st.stop()

# if len(df) < 200:
#     st.warning(f"Only **{len(df)} rows** downloaded. Try a longer period for better results.")

# # ── STEP 2: Price chart ───────────────────────────────────────────
# bar.progress(15, text="📊 Rendering chart...")
# st.subheader(f"📊 {ticker}  —  {period_label} Price History")
# st.plotly_chart(candlestick_fig(df, ticker), use_container_width=True)

# last_close = float(df["Close"].iloc[-1])
# prev_close = float(df["Close"].iloc[-2])
# chg        = (last_close - prev_close) / prev_close * 100
# hi52       = float(df["Close"].tail(252).max())
# lo52       = float(df["Close"].tail(252).min())

# co1, co2, co3, co4 = st.columns(4)
# co1.metric("Last Close",   f"${last_close:.2f}", f"{chg:+.2f}%")
# co2.metric("52-Week High", f"${hi52:.2f}")
# co3.metric("52-Week Low",  f"${lo52:.2f}")
# co4.metric("Trading Days", f"{len(df):,}")
# st.markdown("---")

# # ── STEP 3: Feature engineering ───────────────────────────────────
# bar.progress(25, text="⚙️  Engineering features...")
# features_df  = engineer_features(df)
# feature_cols = features_df.columns.tolist()
# n_features   = len(feature_cols)

# with st.expander(f"🔬 Engineered features ({n_features} columns) — last 10 rows"):
#     st.dataframe(features_df.tail(10).style.format("{:.4f}"),
#                  use_container_width=True)

# # ── STEP 4: Labels ────────────────────────────────────────────────
# bar.progress(30, text="🏷️  Creating BUY/SELL labels...")
# labels           = create_labels(df, features_df.index, forecast_days, return_threshold)
# labels           = labels.iloc[:-forecast_days]
# features_trimmed = features_df.iloc[:-forecast_days]

# l1, l2 = st.columns(2)
# l1.metric("BUY  labels",  f"{labels.sum():,} ({labels.mean()*100:.1f}%)")
# l2.metric("SELL labels",  f"{(labels==0).sum():,} ({(1-labels.mean())*100:.1f}%)")

# # ── STEP 5: Scale ─────────────────────────────────────────────────
# bar.progress(35, text="📐 Scaling features...")
# split_train = int(len(features_trimmed) * 0.80)
# scaler      = MinMaxScaler()
# scaler.fit(features_trimmed.values[:split_train])
# features_scaled = scaler.transform(features_trimmed.values)

# # ── STEP 6: Sequences ─────────────────────────────────────────────
# bar.progress(40, text="🪟 Creating sliding window sequences...")
# X, y   = make_sequences(features_scaled, labels.values, window_size)
# trn_n  = int(len(X) * 0.80)
# val_n  = int(len(X) * 0.10)

# X_train, y_train = X[:trn_n],              y[:trn_n]
# X_val,   y_val   = X[trn_n:trn_n+val_n],   y[trn_n:trn_n+val_n]
# X_test,  y_test  = X[trn_n+val_n:],        y[trn_n+val_n:]
# test_dates        = features_trimmed.index[window_size + trn_n + val_n:]

# st.markdown("**Chronological data split (no shuffling):**")
# s1, s2, s3 = st.columns(3)
# s1.metric("Train", f"{len(X_train):,}")
# s2.metric("Val",   f"{len(X_val):,}")
# s3.metric("Test",  f"{len(X_test):,}")
# st.caption(f"LSTM input shape per sample: `({window_size} timesteps × {n_features} features)`")

# # ── STEP 7: Build model ───────────────────────────────────────────
# bar.progress(50, text="🏗️  Building LSTM model...")
# model = build_lstm_model(window_size, n_features,
#                          lstm_units_1, lstm_units_2, dropout_rate)

# with st.expander("🏗️ Model Architecture"):
#     lines = []
#     model.summary(print_fn=lambda x: lines.append(x))
#     st.code("\n".join(lines), language="text")

# # ── STEP 8: Train ─────────────────────────────────────────────────
# bar.progress(55, text="🔁 Training LSTM — please wait...")

# history = model.fit(
#     X_train, y_train,
#     epochs=epochs,
#     batch_size=64,
#     validation_data=(X_val, y_val),
#     callbacks=[
#         callbacks.EarlyStopping(monitor="val_loss", patience=12,
#                                 restore_best_weights=True, verbose=0),
#         callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
#                                    patience=6, min_lr=1e-6, verbose=0)
#     ],
#     verbose=0
# )
# actual_epochs = len(history.history["loss"])

# # ── STEP 9: Evaluate ──────────────────────────────────────────────
# bar.progress(85, text="📈 Evaluating...")
# y_prob = model.predict(X_test, verbose=0).flatten()
# y_pred = (y_prob > 0.5).astype(int)
# acc    = accuracy_score(y_test, y_pred)
# try:
#     auc = roc_auc_score(y_test, y_prob)
# except Exception:
#     auc = float("nan")

# # ── STEP 10: Latest signal ────────────────────────────────────────
# bar.progress(95, text="🔮 Generating signal...")
# latest_prob   = float(model.predict(
#     features_scaled[-window_size:].reshape(1, window_size, n_features),
#     verbose=0)[0][0])
# signal_label, signal_type, signal_icon = decode_signal(latest_prob)

# bar.progress(100, text="✅ Done!")
# bar.empty()

# # ══════════════════════════════════════════════════════════════════
# # RESULTS
# # ══════════════════════════════════════════════════════════════════

# st.markdown("---")
# st.subheader("🔮 Investment Decision")

# sig_color = {"buy": "#00ff9d", "hold": "#ffb800", "sell": "#ff4d4d"}[signal_type]
# st.markdown(f"""
# <div style="background:#0a1520;border:2px solid {sig_color};padding:2rem;
#             text-align:center;border-radius:4px;margin-bottom:1.5rem;">
#   <div style="font-size:3.5rem;font-weight:900;color:{sig_color};font-family:monospace;">
#     {signal_icon} {signal_label}
#   </div>
#   <div style="color:#7a9ab5;font-family:monospace;font-size:.85rem;
#               letter-spacing:.1em;margin-top:.5rem;">
#     MODEL BUY PROBABILITY: {latest_prob*100:.1f}%
#     &nbsp;|&nbsp; TICKER: {ticker}
#     &nbsp;|&nbsp; FORECAST: {forecast_days} TRADING DAYS
#   </div>
# </div>
# """, unsafe_allow_html=True)

# advice = {
#     "buy": f"""
# **Why BUY?** The LSTM detected a **bullish candlestick pattern** over the last {window_size} trading days of **{ticker}**.
# The model gives a **{latest_prob*100:.1f}% probability** that the price will rise more than {return_threshold*100:.1f}%
# in the next {forecast_days} trading days.

# **Suggested action:** Consider entering a position. Place a stop-loss ~2% below entry price. Watch for confirmation
# via increasing volume on up-candles.
# """,
#     "hold": f"""
# **Why HOLD?** The LSTM sees a **mixed, sideways pattern** in **{ticker}**. BUY probability is {latest_prob*100:.1f}% —
# not strong enough to commit capital, not weak enough to short/exit.

# **Suggested action:** Stay on the sidelines. Wait for a cleaner breakout in either direction. Re-run after the
# next major earnings release or economic event.
# """,
#     "sell": f"""
# **Why SELL / AVOID?** The LSTM detected a **bearish pattern** in **{ticker}**. BUY probability is only {latest_prob*100:.1f}%,
# meaning the model leans {(1-latest_prob)*100:.1f}% bearish for the next {forecast_days} trading days.

# **Suggested action:** Avoid new positions. If already holding, consider tightening your stop-loss. Wait for a
# confirmed bullish reversal (e.g., hammer candle + volume spike) before re-entering.
# """
# }[signal_type]

# st.markdown(advice)
# st.warning("⚠️ **Disclaimer:** For educational purposes only. Never make real investment decisions based solely on "
#            "ML signals. Always consult a certified financial advisor.")

# st.markdown("---")
# st.subheader("📊 Model Performance on Held-Out Test Data")

# p1, p2, p3, p4 = st.columns(4)
# p1.metric("Test Accuracy",  f"{acc*100:.1f}%")
# p2.metric("ROC-AUC",        f"{auc:.3f}" if not math.isnan(auc) else "N/A")
# p3.metric("Epochs Trained", str(actual_epochs))
# p4.metric("Test Samples",   f"{len(X_test):,}")

# st.caption("**Accuracy** = % of days correctly predicted.  "
#            "**ROC-AUC** above 0.55 is meaningful for financial data (0.5 = random, 1.0 = perfect).")

# with st.expander("📋 Full Classification Report"):
#     rpt = classification_report(y_test, y_pred,
#                                 target_names=["SELL/HOLD", "BUY"],
#                                 output_dict=True)
#     st.dataframe(pd.DataFrame(rpt).T.style.format("{:.3f}"),
#                  use_container_width=True)

# st.markdown("---")

# col_l, col_r = st.columns(2)
# with col_l:
#     st.subheader("📉 Training Loss Curve")
#     st.plotly_chart(loss_fig(history), use_container_width=True)
#     st.caption("Validation loss rising while train loss falls = overfitting → increase Dropout.")
# with col_r:
#     st.subheader("📡 BUY Probability — Test Period")
#     st.plotly_chart(signal_history_fig(test_dates[:len(y_prob)], y_prob),
#                     use_container_width=True)
#     st.caption("Green zone >0.65 = BUY. Yellow 0.35–0.65 = HOLD. Red <0.35 = SELL.")

# # ── Signals overlaid on price ─────────────────────────────────────
# st.markdown("---")
# st.subheader("🕯️ BUY / SELL Signals on Price Chart")
# test_close = df["Close"].reindex(test_dates[:len(y_prob)])
# test_close = test_close.dropna()

# if not test_close.empty:
#     buy_idx  = test_close.index[y_pred[:len(test_close)] == 1]
#     sell_idx = test_close.index[y_pred[:len(test_close)] == 0]
#     fig2 = go.Figure()
#     fig2.add_trace(go.Scatter(x=test_close.index, y=test_close.values,
#                               mode="lines", name="Close",
#                               line=dict(color="#c8dff0", width=1.5)))
#     fig2.add_trace(go.Scatter(x=buy_idx, y=test_close.loc[buy_idx].values,
#                               mode="markers", name="BUY",
#                               marker=dict(color="#00ff9d", size=7, symbol="triangle-up")))
#     fig2.add_trace(go.Scatter(x=sell_idx, y=test_close.loc[sell_idx].values,
#                               mode="markers", name="SELL",
#                               marker=dict(color="#ff4d4d", size=7, symbol="triangle-down")))
#     fig2.update_layout(plot_bgcolor="#050a0f", paper_bgcolor="#050a0f",
#                        font_color="#c8dff0", height=380,
#                        legend=dict(bgcolor="#0a1520"),
#                        margin=dict(l=10, r=10, t=20, b=10))
#     fig2.update_xaxes(gridcolor="#0f2035"); fig2.update_yaxes(gridcolor="#0f2035")
#     st.plotly_chart(fig2, use_container_width=True)
#     st.caption("🔺 Green = model said BUY on that day.  🔻 Red = model said SELL/HOLD.")

# st.markdown("---")
# st.subheader("🔧 Tuning Guide")
# st.markdown("""
# | Parameter | Current | Effect if increased |
# |---|---|---|
# | Lookback Window | """ + str(window_size) + """ days | More historical context; slower training |
# | LSTM Layer 1 Units | """ + str(lstm_units_1) + """ | More capacity to learn patterns; risk of overfitting |
# | LSTM Layer 2 Units | """ + str(lstm_units_2) + """ | Same as above |
# | Dropout Rate | """ + str(dropout_rate) + """ | More regularization; use 0.3–0.4 if overfitting |
# | Forecast Horizon | """ + str(forecast_days) + """ days | Longer = harder to predict but more strategic |
# | BUY Threshold | """ + f"{return_threshold*100:.1f}%" + """ | Higher = fewer but higher-conviction BUY signals |
# """)

# st.markdown("---")
# st.caption("Built with `tf.keras.layers.LSTM` · `tf.keras.layers.GRU` · "
#            "`yfinance` · `Streamlit` · `Plotly`")





import math
import datetime
import warnings
import io
import base64

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score, confusion_matrix)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ── yfinance import — show clear error message if version is too old ──
try:
    import yfinance as yf
    _yf_ver = tuple(int(x) for x in yf.__version__.split(".")[:3])
    if _yf_ver < (0, 2, 40):
        st.warning(
            f"You have yfinance **{yf.__version__}** — this version is too old.\n\n"
            "Fix by running this in your terminal, then restart Streamlit:\n"
            "```\npip uninstall yfinance multitasking -y\n"
            'pip install "yfinance>=0.2.40"\n```'
        )
except TypeError:
    st.error(
        "yfinance import failed.\n\n"
        "Fix:\n```\npip uninstall yfinance multitasking -y\n"
        'pip install "yfinance>=0.2.40"\n```'
    )
    st.stop()
except Exception as e:
    st.error(f"Cannot import yfinance: {e}")
    st.stop()

# ── smtplib for sending email alerts ─────────────────────────────
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ═════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="LSTM Stock Analyzer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme state — dark mode is enabled by default ────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# Set color variables based on selected theme
if st.session_state.dark_mode:
    BG      = "#050a0f"
    SIDEBAR = "#080f18"
    CARD    = "#0a1520"
    BORDER  = "#1a3050"
    TEXT    = "#c8dff0"
    SUBTEXT = "#7a9ab5"
    ACCENT  = "#00e5ff"
    PLOT_BG = "#050a0f"
    GRID    = "#0f2035"
else:
    BG      = "#f0f4f8"
    SIDEBAR = "#e2eaf2"
    CARD    = "#ffffff"
    BORDER  = "#b0c8e0"
    TEXT    = "#1a2a3a"
    SUBTEXT = "#4a6a8a"
    ACCENT  = "#0077aa"
    PLOT_BG = "#ffffff"
    GRID    = "#d0dce8"

# Apply CSS styles based on the current theme
st.markdown(f"""
<style>
  [data-testid="stAppViewContainer"] {{ background:{BG}; }}
  [data-testid="stSidebar"]          {{ background:{SIDEBAR}; border-right:1px solid {BORDER}; }}
  .stButton > button {{
    background:{ACCENT}15; border:1px solid {ACCENT};
    color:{ACCENT}; font-family:monospace; letter-spacing:.06em; transition:all .2s;
  }}
  .stButton > button:hover {{ background:{ACCENT}30; }}
  h1,h2,h3 {{ color:{TEXT} !important; }}
  .stProgress > div > div {{ background:linear-gradient(90deg,{ACCENT},#00ff9d) !important; }}
  code {{ color:{ACCENT} !important; background:{CARD} !important; }}
  [data-testid="stMarkdownContainer"] p {{ color:{TEXT}; }}
  div[data-testid="metric-container"] {{
    background:{CARD}; border:1px solid {BORDER};
    padding:.75rem; border-radius:4px;
  }}
  @media (max-width:768px) {{
    [data-testid="stAppViewContainer"] {{ padding:0.5rem !important; }}
    .stColumns {{ flex-direction:column !important; }}
  }}
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════
# TICKER CATALOGUE — dropdown list of all supported assets
# ═════════════════════════════════════════════════════════════════
TICKERS = {
    # ── US Index ETFs ─────────────────────────────────────────────
    "SPY  — S&P 500 ETF (top 500 US companies)":              "SPY",
    "QQQ  — NASDAQ 100 ETF (tech-heavy)":                     "QQQ",
    "DIA  — Dow Jones Industrial Average ETF":                "DIA",
    "IWM  — Russell 2000 Small-Cap ETF":                      "IWM",
    "VTI  — Total US Stock Market ETF":                       "VTI",
    "VOO  — Vanguard S&P 500 ETF":                            "VOO",
    # ── Sector ETFs ───────────────────────────────────────────────
    "XLK  — Technology Sector ETF":                           "XLK",
    "XLF  — Financial Sector ETF":                            "XLF",
    "XLE  — Energy Sector ETF":                               "XLE",
    "XLV  — Healthcare Sector ETF":                           "XLV",
    "XLY  — Consumer Discretionary ETF":                      "XLY",
    # ── Big Tech Stocks ───────────────────────────────────────────
    "AAPL — Apple Inc.":                                      "AAPL",
    "MSFT — Microsoft Corporation":                           "MSFT",
    "GOOGL— Alphabet (Google)":                               "GOOGL",
    "AMZN — Amazon.com Inc.":                                 "AMZN",
    "NVDA — NVIDIA Corporation":                              "NVDA",
    "META — Meta Platforms (Facebook)":                       "META",
    "TSLA — Tesla Inc. (high volatility)":                    "TSLA",
    # ── Pakistan Stock Exchange (PSX) ─────────────────────────────
    "OGDC.KA — Oil & Gas Dev Co (PSX)":                      "OGDC.KA",
    "HBL.KA  — Habib Bank Limited (PSX)":                    "HBL.KA",
    "LUCK.KA — Lucky Cement (PSX)":                          "LUCK.KA",
    "PSO.KA  — Pakistan State Oil (PSX)":                    "PSO.KA",
    "ENGRO.KA— Engro Corporation (PSX)":                     "ENGRO.KA",
    "UBL.KA  — United Bank Limited (PSX)":                   "UBL.KA",
    "MCB.KA  — MCB Bank (PSX)":                              "MCB.KA",
    # ── Forex Currency Pairs ──────────────────────────────────────
    "PKR=X   — USD/PKR (US Dollar to Pakistani Rupee)":      "PKR=X",
    "EURUSD=X— EUR/USD (Euro to US Dollar)":                 "EURUSD=X",
    "GBPUSD=X— GBP/USD (British Pound to US Dollar)":        "GBPUSD=X",
    "JPYUSD=X— JPY/USD (Japanese Yen to US Dollar)":         "JPYUSD=X",
    # ── Commodities & Other Assets ────────────────────────────────
    "GLD  — Gold ETF":                                        "GLD",
    "SLV  — Silver ETF":                                      "SLV",
    "USO  — Oil ETF (WTI Crude)":                             "USO",
    "TLT  — US Treasury 20yr Bond ETF":                       "TLT",
    "BTC-USD — Bitcoin (USD)":                                "BTC-USD",
    "ETH-USD — Ethereum (USD)":                               "ETH-USD",
}

# ═════════════════════════════════════════════════════════════════
# SIDEBAR — all user controls and settings
# ═════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### LSTM Fund Analyzer Pro")

    # Dark / Light mode toggle button
    mode_label = "☀️ Switch to Light Mode" if st.session_state.dark_mode else "🌙 Switch to Dark Mode"
    if st.button(mode_label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown("---")

    # Asset selection dropdown
    ticker_label = st.selectbox(
        "Select Asset",
        options=list(TICKERS.keys()),
        index=0,
        help="Choose a stock, ETF, PSX stock, Forex pair, or Commodity"
    )
    ticker = TICKERS[ticker_label]
    st.caption(f"Ticker symbol: **`{ticker}`**")

    # Historical data period selection
    period_label = st.selectbox(
        "Historical Data Period",
        options=["1 Year", "2 Years", "5 Years", "10 Years", "Max Available"],
        index=2,
        help="More data means better LSTM training. Recommended: 5 Years minimum."
    )
    period_map = {
        "1 Year": 1, "2 Years": 2, "5 Years": 5,
        "10 Years": 10, "Max Available": 20
    }
    period_years = period_map[period_label]

    # Model input parameters
    window_size      = st.slider(
        "Lookback Window (trading days)", 20, 120, 60, 5,
        help="Number of past days the LSTM looks at before making a prediction"
    )
    forecast_days    = st.slider(
        "Forecast Horizon (days ahead)", 5, 30, 10, 5,
        help="Number of days into the future the model predicts"
    )
    return_threshold = st.slider(
        "BUY Signal Threshold (%)", 0.5, 5.0, 1.5, 0.5,
        help="Minimum expected percentage gain to classify a day as BUY"
    ) / 100.0

    st.markdown("---")
    st.markdown("**Model Architecture Settings**")

    # Model type selection
    model_type    = st.selectbox(
        "Model Type",
        ["Stacked LSTM", "Bidirectional LSTM", "LSTM + GRU"],
        help="Choose the neural network architecture to use"
    )
    lstm_units_1  = st.select_slider("Layer 1 Units", [32, 64, 128, 256], value=128)
    lstm_units_2  = st.select_slider("Layer 2 Units", [16, 32,  64, 128], value=64)
    dropout_rate  = st.slider("Dropout Rate", 0.1, 0.5, 0.3, 0.05)
    use_bn        = st.checkbox(
        "Batch Normalization", value=True,
        help="Stabilizes and speeds up training — recommended to keep ON"
    )
    use_attention = st.checkbox(
        "Attention Mechanism", value=False,
        help="Allows the model to focus on the most important timesteps"
    )
    epochs        = st.slider("Max Training Epochs", 20, 150, 80, 10)

    st.markdown("---")
    st.markdown("**Email Alert Settings**")

    # Email alert configuration
    email_alert = st.checkbox("Enable Email Alert on BUY Signal")
    alert_email = ""
    smtp_user   = ""
    smtp_pass   = ""
    if email_alert:
        alert_email = st.text_input("Your Email Address",   placeholder="you@gmail.com")
        smtp_user   = st.text_input("Gmail Sender Address", placeholder="sender@gmail.com")
        smtp_pass   = st.text_input(
            "Gmail App Password", type="password",
            help="Use a Gmail App Password — not your real Gmail password"
        )

    st.markdown("---")
    run_btn = st.button("🚀  Run Analysis", use_container_width=True)


# ═════════════════════════════════════════════════════════════════
# DATA LOADING
# Uses explicit start/end dates and Ticker().history() for stability
# ═════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_data(ticker: str, years: int) -> pd.DataFrame:
    """
    Download OHLCV price data from Yahoo Finance.

    - Uses explicit start/end dates for reliability
    - Strips timezone from index to avoid tz-aware vs tz-naive errors
    - Flattens MultiIndex columns if present in some yfinance versions
    - Returns only the OHLCV columns needed for analysis
    """
    end_dt   = datetime.date.today()
    start_dt = end_dt - datetime.timedelta(days=365 * years)

    tk = yf.Ticker(ticker)
    df = tk.history(
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,   # automatically adjusts for splits and dividends
        actions=False       # we do not need dividend or split columns
    )

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Strip timezone from index to prevent tz-aware vs tz-naive conflicts
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Keep only the required OHLCV columns
    needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[needed].dropna()
    return df


# ═════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — 23 features total (original 17 + 6 new)
# No look-ahead bias: every value only uses data available up to that day
# ═════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 23 candlestick and technical indicator features.
    All features are calculated without look-ahead bias.
    """
    f = pd.DataFrame(index=df.index)

    # ── Original 17 features ─────────────────────────────────────

    # Candlestick body features
    f["body"]         = df["Close"] - df["Open"]
    f["body_ratio"]   = f["body"].abs() / (df["High"] - df["Low"] + 1e-8)
    f["upper_shadow"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    f["lower_shadow"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
    f["direction"]    = np.sign(f["body"])

    # Price return features
    f["return_1d"]  = df["Close"].pct_change(1)
    f["return_5d"]  = df["Close"].pct_change(5)
    f["return_10d"] = df["Close"].pct_change(10)

    # Moving average trend features
    sma20 = df["Close"].rolling(20).mean()
    sma50 = df["Close"].rolling(50).mean()
    f["price_vs_sma20"] = df["Close"] / (sma20 + 1e-8)
    f["price_vs_sma50"] = df["Close"] / (sma50 + 1e-8)
    f["sma_cross"]      = sma20 / (sma50 + 1e-8)

    # MACD (Moving Average Convergence Divergence)
    ema12       = df["Close"].ewm(span=12, adjust=False).mean()
    ema26       = df["Close"].ewm(span=26, adjust=False).mean()
    f["macd"]   = (ema12 - ema26) / (df["Close"] + 1e-8)

    # Volatility features
    f["volatility_20d"] = f["return_1d"].rolling(20).std()
    f["atr_ratio"]      = (df["High"] - df["Low"]).rolling(14).mean() / (df["Close"] + 1e-8)

    # Volume features
    vol_avg20          = df["Volume"].rolling(20).mean()
    f["volume_ratio"]  = df["Volume"] / (vol_avg20 + 1)
    obv                = (np.sign(f["return_1d"]) * df["Volume"]).cumsum()
    f["obv_ratio"]     = obv / (obv.abs().rolling(20).mean() + 1)

    # RSI (Relative Strength Index) — normalized to range [0, 1]
    delta      = df["Close"].diff()
    gain       = delta.clip(lower=0).rolling(14).mean()
    loss       = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi"]   = (100 - 100 / (1 + gain / (loss + 1e-8))) / 100.0

    # ── NEW Feature 1: Bollinger Bands ───────────────────────────
    bb_mid            = df["Close"].rolling(20).mean()
    bb_std            = df["Close"].rolling(20).std()
    bb_upper          = bb_mid + 2 * bb_std
    bb_lower          = bb_mid - 2 * bb_std
    f["bb_position"]  = (df["Close"] - bb_lower) / (bb_upper - bb_lower + 1e-8)
    f["bb_width"]     = (bb_upper - bb_lower) / (bb_mid + 1e-8)

    # ── NEW Feature 2: Stochastic Oscillator (%K and %D) ─────────
    low14        = df["Low"].rolling(14).min()
    high14       = df["High"].rolling(14).max()
    f["stoch_k"] = (df["Close"] - low14) / (high14 - low14 + 1e-8)
    f["stoch_d"] = f["stoch_k"].rolling(3).mean()   # %D = 3-day SMA of %K

    # ── NEW Feature 3: Williams %R (momentum indicator) ──────────
    f["williams_r"] = (high14 - df["Close"]) / (high14 - low14 + 1e-8)

    # ── NEW Feature 4: CCI (Commodity Channel Index) ─────────────
    tp           = (df["High"] + df["Low"] + df["Close"]) / 3
    tp_sma       = tp.rolling(20).mean()
    tp_mad       = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    f["cci"]     = (tp - tp_sma) / (0.015 * tp_mad + 1e-8) / 200.0   # normalized

    return f.dropna()


# ═════════════════════════════════════════════════════════════════
# ATTENTION LAYER — custom Keras layer
# ═════════════════════════════════════════════════════════════════
class AttentionLayer(layers.Layer):
    """
    Simple additive (Bahdanau-style) attention mechanism over LSTM time-steps.
    Allows the model to assign higher weights to more important time steps.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_W", shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform", trainable=True
        )
        self.b = self.add_weight(
            name="att_b", shape=(input_shape[-1],),
            initializer="zeros", trainable=True
        )
        self.u = self.add_weight(
            name="att_u", shape=(input_shape[-1], 1),
            initializer="glorot_uniform", trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        # x shape: (batch, timesteps, features)
        score = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        alpha = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        return tf.reduce_sum(x * alpha, axis=1)   # output: (batch, features)


# ═════════════════════════════════════════════════════════════════
# MODEL BUILDER — supports 3 architecture types
# ═════════════════════════════════════════════════════════════════
def build_model(window, n_features, units1, units2, drop,
                model_type, use_bn, use_attention):
    """
    Build and compile the neural network model.

    Supported architectures:
    - Stacked LSTM: Two stacked LSTM layers
    - Bidirectional LSTM: Two Bidirectional LSTM layers (reads sequence forward and backward)
    - LSTM + GRU: One LSTM layer followed by one GRU layer

    Optional enhancements:
    - Batch Normalization: Stabilizes and speeds up training
    - Attention Mechanism: Focuses on the most informative timesteps

    Input shape:  (batch_size, window_size, n_features)
    Output shape: (batch_size, 1) — sigmoid probability of a BUY signal
    """
    inp = layers.Input(shape=(window, n_features), name="input")
    x   = inp

    if model_type == "Stacked LSTM":
        x = layers.LSTM(units1, return_sequences=True, name="lstm_1")(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(drop)(x)
        if use_attention:
            x = AttentionLayer(name="attention")(x)
        else:
            x = layers.LSTM(units2, return_sequences=False, name="lstm_2")(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(drop)(x)

    elif model_type == "Bidirectional LSTM":
        x = layers.Bidirectional(
            layers.LSTM(units1, return_sequences=True), name="bilstm_1")(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(drop)(x)
        if use_attention:
            x = AttentionLayer(name="attention")(x)
        else:
            x = layers.Bidirectional(
                layers.LSTM(units2, return_sequences=False), name="bilstm_2")(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(drop)(x)

    elif model_type == "LSTM + GRU":
        x = layers.LSTM(units1, return_sequences=True, name="lstm_1")(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(drop)(x)
        if use_attention:
            x = AttentionLayer(name="attention")(x)
        else:
            x = layers.GRU(units2, return_sequences=False, name="gru_1")(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(drop)(x)

    x   = layers.Dense(32, activation="relu", name="dense")(x)
    x   = layers.Dropout(drop / 2)(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    mdl = models.Model(
        inputs=inp, outputs=out,
        name=f"LSTM_{model_type.replace(' ', '_')}"
    )
    mdl.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return mdl


# ═════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════

def create_labels(df_orig, feat_index, horizon, threshold):
    """Create binary BUY/SELL labels based on future price returns."""
    aligned = df_orig.loc[feat_index, "Close"]
    ret     = aligned.shift(-horizon) / aligned - 1
    return (ret > threshold).astype(int)


def make_sequences(features_arr, labels_arr, window):
    """Create sliding window sequences for LSTM input."""
    X, y = [], []
    for i in range(window, len(features_arr)):
        X.append(features_arr[i - window: i])
        y.append(labels_arr[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def decode_signal(prob):
    """Convert BUY probability into a human-readable signal label."""
    if prob >= 0.65:    return "STRONG BUY",  "buy",  "🟢"
    elif prob >= 0.52:  return "BUY",          "buy",  "🟩"
    elif prob >= 0.35:  return "HOLD / WAIT",  "hold", "🟡"
    elif prob >= 0.20:  return "SELL",         "sell", "🟥"
    else:               return "STRONG SELL",  "sell", "🔴"


def price_targets(last_close, signal_type, atr):
    """
    Calculate entry price, stop-loss, and two profit targets
    based on the ATR (Average True Range).
    """
    if signal_type == "buy":
        entry     = last_close
        stop_loss = round(last_close - 1.5 * atr, 2)
        target1   = round(last_close + 2.0 * atr, 2)
        target2   = round(last_close + 3.5 * atr, 2)
    else:
        entry     = last_close
        stop_loss = round(last_close + 1.5 * atr, 2)
        target1   = round(last_close - 2.0 * atr, 2)
        target2   = round(last_close - 3.5 * atr, 2)
    return entry, stop_loss, target1, target2


def send_email_alert(to_addr, from_addr, password, ticker, signal, prob, close):
    """Send a BUY signal alert email via Gmail SMTP."""
    try:
        msg            = MIMEMultipart("alternative")
        msg["Subject"] = f"LSTM Signal Alert: {signal} — {ticker}"
        msg["From"]    = from_addr
        msg["To"]      = to_addr
        body = f"""
        <h2 style='color:#00cc66'>LSTM Stock Analyzer — Signal Alert</h2>
        <p><b>Ticker:</b> {ticker}</p>
        <p><b>Signal:</b> {signal}</p>
        <p><b>BUY Probability:</b> {prob * 100:.1f}%</p>
        <p><b>Last Close Price:</b> ${close:.2f}</p>
        <hr>
        <p style='color:gray; font-size:11px'>
        This alert is for educational purposes only. Not financial advice.
        </p>
        """
        msg.attach(MIMEText(body, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(from_addr, password)
            srv.sendmail(from_addr, to_addr, msg.as_string())
        return True
    except Exception as e:
        return str(e)


# ═════════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# ═════════════════════════════════════════════════════════════════

def candlestick_fig(df, ticker):
    """
    Main price chart with candlesticks, volume bars,
    SMA 20, SMA 50, and Bollinger Bands overlaid.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03
    )

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#00ff9d",
        decreasing_line_color="#ff4d4d",
        name=ticker
    ), row=1, col=1)

    # Volume bars
    colors = ["#00ff9d" if c >= o else "#ff4d4d"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=colors, opacity=0.5, name="Volume"
    ), row=2, col=1)

    # SMA 20 line
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"].rolling(20).mean(),
        line=dict(color="#00e5ff", width=1.2), name="SMA 20"
    ), row=1, col=1)

    # SMA 50 line
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"].rolling(50).mean(),
        line=dict(color="#b06cff", width=1.2), name="SMA 50"
    ), row=1, col=1)

    # Bollinger Bands
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    fig.add_trace(go.Scatter(
        x=df.index, y=bb_mid + 2 * bb_std,
        line=dict(color="#ffb800", width=0.8, dash="dot"),
        name="BB Upper"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=bb_mid - 2 * bb_std,
        line=dict(color="#ffb800", width=0.8, dash="dot"),
        fill="tonexty", fillcolor="rgba(255,184,0,0.04)",
        name="BB Lower"
    ), row=1, col=1)

    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        font_color=TEXT, height=520,
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def loss_fig(history):
    """Plot training and validation loss curves across epochs."""
    ep  = list(range(1, len(history.history["loss"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ep, y=history.history["loss"],
        name="Training Loss", line=dict(color="#00e5ff", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=ep, y=history.history["val_loss"],
        name="Validation Loss", line=dict(color="#ffb800", width=2, dash="dash")
    ))
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        font_color=TEXT, height=300,
        xaxis_title="Epoch", yaxis_title="Loss",
        legend=dict(bgcolor=CARD),
        margin=dict(l=10, r=10, t=20, b=10)
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def signal_history_fig(dates, probs):
    """Plot BUY probability over the test period with color-coded zones."""
    fig = go.Figure()
    fig.add_hrect(y0=0.65, y1=1.0, fillcolor="#00ff9d", opacity=0.07,
                  annotation_text="BUY zone",  annotation_font_color="#00ff9d")
    fig.add_hrect(y0=0.35, y1=0.65, fillcolor="#ffb800", opacity=0.05,
                  annotation_text="HOLD zone", annotation_font_color="#ffb800")
    fig.add_hrect(y0=0.0,  y1=0.35, fillcolor="#ff4d4d", opacity=0.07,
                  annotation_text="SELL zone", annotation_font_color="#ff4d4d")
    fig.add_trace(go.Scatter(
        x=dates, y=probs, mode="lines",
        name="BUY Probability",
        line=dict(color="#00e5ff", width=2),
        fill="tozeroy", fillcolor="rgba(0,229,255,0.05)"
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color="#ffffff", opacity=0.3)
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        font_color=TEXT, height=280,
        yaxis=dict(range=[0, 1], title="BUY Probability"),
        margin=dict(l=10, r=10, t=20, b=10)
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def confusion_matrix_fig(y_test, y_pred):
    """Plot a heatmap of the confusion matrix for BUY vs SELL/HOLD predictions."""
    cm     = confusion_matrix(y_test, y_pred)
    labels = ["SELL/HOLD", "BUY"]
    fig    = ff.create_annotated_heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, "#0a1520"], [1, "#00e5ff"]],
        showscale=True
    )
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        font_color=TEXT, height=320,
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text="Confusion Matrix", font=dict(color=TEXT))
    )
    return fig


def feature_importance_fig(feature_cols, features_scaled, model, window):
    """
    Calculate permutation-based feature importance.
    Each feature is shuffled one at a time, and the drop in BUY probability
    is measured. A larger drop means that feature is more important.
    """
    baseline_prob = model.predict(
        features_scaled[-window:].reshape(1, window, len(feature_cols)),
        verbose=0
    )[0][0]

    importances = []
    for i in range(len(feature_cols)):
        perturbed       = features_scaled[-window:].copy()
        perturbed[:, i] = np.random.permutation(perturbed[:, i])
        p = model.predict(
            perturbed.reshape(1, window, len(feature_cols)), verbose=0
        )[0][0]
        importances.append(abs(baseline_prob - p))

    imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    imp_df = imp_df.sort_values("Importance", ascending=True).tail(15)

    fig = go.Figure(go.Bar(
        x=imp_df["Importance"],
        y=imp_df["Feature"],
        orientation="h",
        marker=dict(
            color=imp_df["Importance"],
            colorscale=[[0, "#0a3050"], [1, "#00e5ff"]]
        )
    ))
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        font_color=TEXT, height=420,
        xaxis_title="Impact on BUY Probability (higher = more important)",
        margin=dict(l=10, r=10, t=20, b=10)
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def monte_carlo_fig(last_close, daily_ret_mean, daily_ret_std, days=30, sims=200):
    """
    Simulate future price paths using Monte Carlo method.
    Each path is generated by sampling daily returns from a normal distribution
    based on the asset's historical mean and standard deviation.
    """
    fig   = go.Figure()
    paths = []

    for _ in range(sims):
        shocks = np.random.normal(daily_ret_mean, daily_ret_std, days)
        price  = [last_close]
        for s in shocks:
            price.append(price[-1] * (1 + s))
        paths.append(price)
        fig.add_trace(go.Scatter(
            y=price, mode="lines",
            line=dict(color="rgba(0,229,255,0.06)", width=1),
            showlegend=False
        ))

    paths_arr = np.array(paths)

    # Mean path
    fig.add_trace(go.Scatter(
        y=paths_arr.mean(axis=0), mode="lines",
        line=dict(color="#00ff9d", width=2.5), name="Mean Path"
    ))
    # 5th percentile (pessimistic scenario)
    fig.add_trace(go.Scatter(
        y=np.percentile(paths_arr, 5, axis=0), mode="lines",
        line=dict(color="#ff4d4d", width=1.5, dash="dash"), name="5th Percentile"
    ))
    # 95th percentile (optimistic scenario)
    fig.add_trace(go.Scatter(
        y=np.percentile(paths_arr, 95, axis=0), mode="lines",
        line=dict(color="#00ff9d", width=1.5, dash="dash"), name="95th Percentile"
    ))

    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        font_color=TEXT, height=350,
        xaxis_title=f"Days Ahead (next {days} trading days)",
        yaxis_title="Simulated Price",
        legend=dict(bgcolor=CARD),
        margin=dict(l=10, r=10, t=20, b=10),
        title=dict(text=f"Monte Carlo Simulation ({sims} paths)", font=dict(color=TEXT))
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def backtest_fig(test_close, y_pred_test, initial_capital=10000):
    """
    Compare LSTM strategy returns against a simple Buy & Hold strategy.

    LSTM Strategy logic:
    - On days with a BUY signal: stay invested and earn market returns
    - On days with HOLD/SELL signal: hold cash, earn zero return

    Returns the chart figure plus key performance metrics.
    """
    prices = test_close.values
    n      = min(len(prices), len(y_pred_test))
    prices = prices[:n]
    sigs   = y_pred_test[:n]

    # Calculate daily returns
    rets = np.diff(prices) / prices[:-1]

    # Build Buy & Hold portfolio values
    bh_val = [initial_capital]
    for r in rets:
        bh_val.append(bh_val[-1] * (1 + r))

    # Build LSTM Strategy portfolio values
    lstm_val = [initial_capital]
    for i, r in enumerate(rets):
        if sigs[i] == 1:          # BUY signal — stay in market
            lstm_val.append(lstm_val[-1] * (1 + r))
        else:                      # HOLD/SELL signal — hold cash
            lstm_val.append(lstm_val[-1])

    dates = test_close.index[:n]

    # Plot both portfolios
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=bh_val[:n],
        name="Buy & Hold Strategy",
        line=dict(color="#b06cff", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=lstm_val[:n],
        name="LSTM Strategy",
        line=dict(color="#00ff9d", width=2)
    ))
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        font_color=TEXT, height=350,
        yaxis_title="Portfolio Value (USD)",
        legend=dict(bgcolor=CARD),
        margin=dict(l=10, r=10, t=20, b=10),
        title=dict(
            text=f"Backtest — Starting Capital: ${initial_capital:,}",
            font=dict(color=TEXT)
        )
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)

    # Calculate performance metrics
    bh_ret   = (bh_val[-1] / initial_capital - 1) * 100
    lstm_ret = (lstm_val[-1] / initial_capital - 1) * 100

    # Annualized Sharpe Ratio (assumes 252 trading days per year)
    lstm_daily = np.diff(lstm_val) / np.array(lstm_val[:-1])
    sharpe     = (lstm_daily.mean() / (lstm_daily.std() + 1e-8)) * np.sqrt(252)

    # Maximum Drawdown
    peak   = np.maximum.accumulate(lstm_val)
    dd     = (np.array(lstm_val) - peak) / (peak + 1e-8)
    max_dd = dd.min() * 100

    # Win Rate — percentage of BUY signal days that were actually profitable
    wins   = sum(1 for i, r in enumerate(rets) if sigs[i] == 1 and r > 0)
    buys   = sum(sigs == 1)
    win_rt = (wins / buys * 100) if buys > 0 else 0

    return fig, bh_ret, lstm_ret, sharpe, max_dd, win_rt


# ═════════════════════════════════════════════════════════════════
# LANDING PAGE — shown before the user clicks Run Analysis
# ═════════════════════════════════════════════════════════════════
st.title("📈 LSTM Stock Fund Investment Analyzer Pro")
st.caption(
    "Enhanced with **Bollinger Bands · Stochastic Oscillator · Williams %R · CCI · "
    "Bidirectional LSTM · Attention Mechanism · Backtesting · Monte Carlo Simulation · "
    "PSX / Forex / Commodities support**"
)
st.markdown("---")

if not run_btn:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### How It Works")
        st.markdown("""
1. Downloads real OHLCV price data via Yahoo Finance
2. Engineers **23** candlestick and technical features
3. Trains a 2-layer LSTM / GRU / BiLSTM classifier
4. Outputs BUY / HOLD / SELL signal with confidence %
5. Backtests the strategy against Buy & Hold
6. Runs Monte Carlo simulation for future price range
        """)
    with c2:
        st.markdown("#### Available Model Types")
        st.code("""# Stacked LSTM
LSTM(128) → LSTM(64) → Dense(1)

# Bidirectional LSTM
BiLSTM(128) → BiLSTM(64) → Dense(1)

# LSTM + GRU Hybrid
LSTM(128) → GRU(64) → Dense(1)

# Optional add-ons:
# + Attention Mechanism
# + Batch Normalization""", language="python")
    with c3:
        st.markdown("#### Required Installations")
        st.code("""pip uninstall yfinance multitasking -y
pip install "yfinance>=0.2.40"
pip install streamlit plotly
pip install scikit-learn tensorflow""", language="bash")

    st.info("Select an asset from the sidebar on the left and click **🚀 Run Analysis** to begin.")
    st.stop()


# ═════════════════════════════════════════════════════════════════
# MAIN PIPELINE — runs after the user clicks Run Analysis
# ═════════════════════════════════════════════════════════════════
bar = st.progress(0, text="")

# ── Step 1: Download price data ───────────────────────────────────
bar.progress(5, text=f"Downloading {ticker} data ({period_label})...")
try:
    df = load_data(ticker, period_years)
except Exception as e:
    st.error(
        f"Download failed for **{ticker}**: {e}\n\n"
        "Make sure you have run:\n"
        "```\npip uninstall yfinance multitasking -y\n"
        'pip install "yfinance>=0.2.40"\n```'
    )
    st.stop()

if df is None or df.empty:
    st.error(f"**`{ticker}`** returned empty data. Please try a different asset or period.")
    st.stop()

if len(df) < 200:
    st.warning(f"Only **{len(df)} rows** of data available. A longer period will give better results.")

# ── Step 2: Render price chart ────────────────────────────────────
bar.progress(15, text="Rendering price chart...")
st.subheader(f"📊 {ticker}  —  {period_label} Price History")
st.plotly_chart(candlestick_fig(df, ticker), use_container_width=True)

last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2])
chg        = (last_close - prev_close) / prev_close * 100
hi52       = float(df["Close"].tail(252).max())
lo52       = float(df["Close"].tail(252).min())

co1, co2, co3, co4 = st.columns(4)
co1.metric("Last Close Price", f"${last_close:.2f}", f"{chg:+.2f}%")
co2.metric("52-Week High",     f"${hi52:.2f}")
co3.metric("52-Week Low",      f"${lo52:.2f}")
co4.metric("Total Trading Days in Data", f"{len(df):,}")
st.markdown("---")

# ── Step 3: Engineer features ─────────────────────────────────────
bar.progress(25, text="Calculating 23 technical features...")
features_df  = engineer_features(df)
feature_cols = features_df.columns.tolist()
n_features   = len(feature_cols)

with st.expander(f"View Engineered Features ({n_features} columns) — Last 10 Rows"):
    st.dataframe(features_df.tail(10).style.format("{:.4f}"), use_container_width=True)

# ── Step 4: Create BUY/SELL labels ───────────────────────────────
bar.progress(30, text="Creating BUY / SELL labels...")
labels           = create_labels(df, features_df.index, forecast_days, return_threshold)
labels           = labels.iloc[:-forecast_days]
features_trimmed = features_df.iloc[:-forecast_days]

l1, l2 = st.columns(2)
l1.metric("BUY Labels",       f"{labels.sum():,} ({labels.mean() * 100:.1f}%)")
l2.metric("SELL/HOLD Labels", f"{(labels == 0).sum():,} ({(1 - labels.mean()) * 100:.1f}%)")

# ── Step 5: Scale features ────────────────────────────────────────
bar.progress(35, text="Scaling features with MinMaxScaler...")
split_train     = int(len(features_trimmed) * 0.80)
scaler          = MinMaxScaler()
scaler.fit(features_trimmed.values[:split_train])
features_scaled = scaler.transform(features_trimmed.values)

# ── Step 6: Create sliding window sequences ───────────────────────
bar.progress(40, text="Creating sliding window sequences...")
X, y  = make_sequences(features_scaled, labels.values, window_size)
trn_n = int(len(X) * 0.80)
val_n = int(len(X) * 0.10)

X_train, y_train = X[:trn_n],              y[:trn_n]
X_val,   y_val   = X[trn_n:trn_n + val_n], y[trn_n:trn_n + val_n]
X_test,  y_test  = X[trn_n + val_n:],      y[trn_n + val_n:]
test_dates        = features_trimmed.index[window_size + trn_n + val_n:]

st.markdown("**Chronological Data Split (no random shuffling — preserves time order):**")
s1, s2, s3 = st.columns(3)
s1.metric("Training Samples",   f"{len(X_train):,}")
s2.metric("Validation Samples", f"{len(X_val):,}")
s3.metric("Test Samples",       f"{len(X_test):,}")
st.caption(f"LSTM input shape per sample: `({window_size} timesteps × {n_features} features)`")

# ── Step 7: Build the model ───────────────────────────────────────
bar.progress(50, text="Building neural network model...")
model = build_model(
    window_size, n_features, lstm_units_1, lstm_units_2,
    dropout_rate, model_type, use_bn, use_attention
)

with st.expander("View Model Architecture Summary"):
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    st.code("\n".join(lines), language="text")

# ── Step 8: Train the model ───────────────────────────────────────
bar.progress(55, text="Training model — this may take a few minutes...")
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[
        callbacks.EarlyStopping(
            monitor="val_loss", patience=12,
            restore_best_weights=True, verbose=0
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=6, min_lr=1e-6, verbose=0
        )
    ],
    verbose=0
)
actual_epochs = len(history.history["loss"])

# ── Step 9: Evaluate on test set ──────────────────────────────────
bar.progress(85, text="Evaluating model on test data...")
y_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_prob > 0.5).astype(int)
acc    = accuracy_score(y_test, y_pred)
try:
    auc = roc_auc_score(y_test, y_prob)
except Exception:
    auc = float("nan")

# ── Step 10: Generate latest signal ──────────────────────────────
bar.progress(95, text="Generating latest investment signal...")
latest_prob = float(model.predict(
    features_scaled[-window_size:].reshape(1, window_size, n_features),
    verbose=0
)[0][0])
signal_label, signal_type, signal_icon = decode_signal(latest_prob)

# Calculate ATR for price target calculation
atr_val = float((df["High"] - df["Low"]).rolling(14).mean().iloc[-1])

bar.progress(100, text="Analysis complete!")
bar.empty()


# ═════════════════════════════════════════════════════════════════
# RESULTS SECTION
# ═════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("🔮 Investment Decision")

# Main signal display card
sig_color = {"buy": "#00ff9d", "hold": "#ffb800", "sell": "#ff4d4d"}[signal_type]
st.markdown(f"""
<div style="background:{CARD}; border:2px solid {sig_color}; padding:2rem;
            text-align:center; border-radius:4px; margin-bottom:1.5rem;">
  <div style="font-size:3.5rem; font-weight:900; color:{sig_color}; font-family:monospace;">
    {signal_icon} {signal_label}
  </div>
  <div style="color:{SUBTEXT}; font-family:monospace; font-size:.85rem;
              letter-spacing:.1em; margin-top:.5rem;">
    BUY PROBABILITY: {latest_prob * 100:.1f}%
    &nbsp;|&nbsp; TICKER: {ticker}
    &nbsp;|&nbsp; MODEL: {model_type}
    &nbsp;|&nbsp; FORECAST: {forecast_days} TRADING DAYS
  </div>
</div>
""", unsafe_allow_html=True)

# ── Price Target Calculator ───────────────────────────────────────
entry, stop_loss, target1, target2 = price_targets(last_close, signal_type, atr_val)
st.markdown("#### 🎯 Price Target Calculator (based on ATR)")
pt1, pt2, pt3, pt4 = st.columns(4)
pt1.metric("Entry Price",            f"${entry:.2f}")
pt2.metric("Stop Loss",              f"${stop_loss:.2f}",
           f"{(stop_loss - entry) / entry * 100:+.1f}%")
pt3.metric("Profit Target 1 (2×ATR)", f"${target1:.2f}",
           f"{(target1 - entry) / entry * 100:+.1f}%")
pt4.metric("Profit Target 2 (3.5×ATR)", f"${target2:.2f}",
           f"{(target2 - entry) / entry * 100:+.1f}%")

# Signal explanation text
advice = {
    "buy": f"""
**Why BUY?** The {model_type} model detected a **bullish pattern** over the last {window_size} trading days of **{ticker}**.
The model gives a **{latest_prob * 100:.1f}% probability** that the price will rise more than {return_threshold * 100:.1f}%
in the next {forecast_days} trading days.

**Suggested action:** Consider entering near **${entry:.2f}**. Place a stop-loss at **${stop_loss:.2f}** (approximately 1.5×ATR below entry).
Conservative target: **${target1:.2f}** — Aggressive target: **${target2:.2f}**.
""",
    "hold": f"""
**Why HOLD?** The {model_type} model sees a **mixed, sideways pattern** in **{ticker}**.
BUY probability is {latest_prob * 100:.1f}% — not strong enough to enter, not weak enough to exit.

**Suggested action:** Stay on the sidelines. Wait for a clearer directional breakout before committing capital.
Re-run the analysis after the next major earnings release or economic event.
""",
    "sell": f"""
**Why SELL / AVOID?** The {model_type} model detected a **bearish pattern** in **{ticker}**.
BUY probability is only {latest_prob * 100:.1f}%, meaning the model leans {(1 - latest_prob) * 100:.1f}% bearish
for the next {forecast_days} trading days.

**Suggested action:** Avoid opening new long positions. If already holding, consider tightening your stop-loss
to around **${stop_loss:.2f}**. Wait for a confirmed bullish reversal signal before re-entering.
"""
}[signal_type]

st.markdown(advice)
st.warning(
    "**Disclaimer:** This tool is for educational and learning purposes only. "
    "Never make real investment decisions based solely on machine learning signals. "
    "Always consult a certified financial advisor before investing."
)

# ── Email Alert ───────────────────────────────────────────────────
if email_alert and signal_type == "buy" and alert_email and smtp_user and smtp_pass:
    result = send_email_alert(
        alert_email, smtp_user, smtp_pass,
        ticker, signal_label, latest_prob, last_close
    )
    if result is True:
        st.success(f"BUY alert email successfully sent to **{alert_email}**!")
    else:
        st.error(f"Email sending failed: {result}")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE METRICS
# ═════════════════════════════════════════════════════════════════
st.subheader("📊 Model Performance on Held-Out Test Data")

p1, p2, p3, p4 = st.columns(4)
p1.metric("Test Accuracy",        f"{acc * 100:.1f}%")
p2.metric("ROC-AUC Score",        f"{auc:.3f}" if not math.isnan(auc) else "N/A")
p3.metric("Epochs Trained",       str(actual_epochs))
p4.metric("Test Set Size",        f"{len(X_test):,} samples")

st.caption(
    "**Accuracy** = percentage of days correctly predicted as BUY or SELL.  "
    "**ROC-AUC above 0.55** is meaningful for financial data (0.5 = random guess, 1.0 = perfect)."
)

with st.expander("View Full Classification Report"):
    rpt = classification_report(
        y_test, y_pred,
        target_names=["SELL/HOLD", "BUY"],
        output_dict=True
    )
    st.dataframe(pd.DataFrame(rpt).T.style.format("{:.3f}"), use_container_width=True)

st.markdown("---")

# ── Training loss curves and BUY probability chart ────────────────
col_l, col_r = st.columns(2)
with col_l:
    st.subheader("📉 Training Loss Curve")
    st.plotly_chart(loss_fig(history), use_container_width=True)
    st.caption(
        "If validation loss rises while training loss keeps falling, the model is overfitting — "
        "try increasing the dropout rate."
    )
with col_r:
    st.subheader("📡 BUY Probability Over Test Period")
    st.plotly_chart(signal_history_fig(test_dates[:len(y_prob)], y_prob),
                    use_container_width=True)
    st.caption("Green zone (>0.65) = BUY. Yellow (0.35–0.65) = HOLD. Red (<0.35) = SELL.")

st.markdown("---")

# ── Confusion Matrix ──────────────────────────────────────────────
st.subheader("🔲 Confusion Matrix")
st.plotly_chart(confusion_matrix_fig(y_test, y_pred), use_container_width=True)
st.caption(
    "Rows represent the actual label. Columns represent the predicted label. "
    "Diagonal cells are correct predictions."
)

st.markdown("---")

# ── Feature Importance ────────────────────────────────────────────
st.subheader("🔍 Feature Importance (Permutation Method)")
with st.spinner("Calculating feature importance — please wait..."):
    fi_fig = feature_importance_fig(feature_cols, features_scaled, model, window_size)
st.plotly_chart(fi_fig, use_container_width=True)
st.caption(
    "A longer bar means that feature has a greater impact on the model's BUY prediction. "
    "Calculated by shuffling each feature and measuring how much the prediction changes."
)

st.markdown("---")

# ── BUY/SELL signals overlaid on price chart ──────────────────────
st.subheader("🕯️ BUY / SELL Signals Overlaid on Price Chart")
test_close = df["Close"].reindex(test_dates[:len(y_prob)])
test_close = test_close.dropna()

if not test_close.empty:
    buy_idx  = test_close.index[y_pred[:len(test_close)] == 1]
    sell_idx = test_close.index[y_pred[:len(test_close)] == 0]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=test_close.index, y=test_close.values,
        mode="lines", name="Close Price",
        line=dict(color="#c8dff0", width=1.5)
    ))
    fig2.add_trace(go.Scatter(
        x=buy_idx, y=test_close.loc[buy_idx].values,
        mode="markers", name="BUY Signal",
        marker=dict(color="#00ff9d", size=7, symbol="triangle-up")
    ))
    fig2.add_trace(go.Scatter(
        x=sell_idx, y=test_close.loc[sell_idx].values,
        mode="markers", name="SELL / HOLD Signal",
        marker=dict(color="#ff4d4d", size=7, symbol="triangle-down")
    ))
    fig2.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        font_color=TEXT, height=380,
        legend=dict(bgcolor=CARD),
        margin=dict(l=10, r=10, t=20, b=10)
    )
    fig2.update_xaxes(gridcolor=GRID)
    fig2.update_yaxes(gridcolor=GRID)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("🔺 Green triangles = model predicted BUY on that day.  🔻 Red triangles = model predicted SELL or HOLD.")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════
# BACKTESTING SECTION
# ═════════════════════════════════════════════════════════════════
st.subheader("💰 Backtest — LSTM Strategy vs Buy & Hold")

if not test_close.empty:
    bt_fig, bh_ret, lstm_ret, sharpe, max_dd, win_rt = backtest_fig(
        test_close, y_pred[:len(test_close)]
    )
    st.plotly_chart(bt_fig, use_container_width=True)

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Buy & Hold Return",    f"{bh_ret:+.1f}%")
    b2.metric("LSTM Strategy Return", f"{lstm_ret:+.1f}%",
              f"{lstm_ret - bh_ret:+.1f}% vs Buy & Hold")
    b3.metric("Annualized Sharpe Ratio", f"{sharpe:.2f}")
    b4.metric("Maximum Drawdown",        f"{max_dd:.1f}%")

    b5, b6 = st.columns(2)
    b5.metric("Win Rate on BUY Signal Days", f"{win_rt:.1f}%")
    b6.metric("Total BUY Signal Days",        f"{int(y_pred[:len(test_close)].sum()):,}")

    st.caption(
        "**Sharpe Ratio > 1** is considered good. **Sharpe Ratio > 2** is excellent.  "
        "**Maximum Drawdown** = the worst peak-to-trough portfolio loss during the test period."
    )

st.markdown("---")

# ═════════════════════════════════════════════════════════════════
# MONTE CARLO SIMULATION
# ═════════════════════════════════════════════════════════════════
st.subheader("🎲 Monte Carlo Future Price Simulation")
daily_returns = df["Close"].pct_change().dropna()
mc_fig = monte_carlo_fig(
    last_close,
    daily_ret_mean=float(daily_returns.mean()),
    daily_ret_std=float(daily_returns.std()),
    days=forecast_days * 3,
    sims=300
)
st.plotly_chart(mc_fig, use_container_width=True)
st.caption(
    f"300 randomly simulated price paths based on **{ticker}**'s historical daily return distribution.  "
    "Solid green line = mean (average) path.  Dashed lines = 5th and 95th percentile boundaries."
)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════
# EXPORT RESULTS AS CSV
# ═════════════════════════════════════════════════════════════════
st.subheader("💾 Export Results")

export_df = pd.DataFrame({
    "Date":            test_dates[:len(y_prob)],
    "Close_Price":     df["Close"].reindex(test_dates[:len(y_prob)]).values,
    "BUY_Probability": y_prob,
    "Signal":          [
        "BUY" if p >= 0.52 else "HOLD" if p >= 0.35 else "SELL"
        for p in y_prob
    ]
})

csv_buf = io.StringIO()
export_df.to_csv(csv_buf, index=False)

st.download_button(
    label="⬇️  Download Signals as CSV",
    data=csv_buf.getvalue(),
    file_name=f"{ticker}_lstm_signals_{datetime.date.today()}.csv",
    mime="text/csv"
)
st.caption(
    "The CSV file contains the date, closing price, BUY probability, "
    "and predicted signal for every day in the test period."
)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════
# PARAMETER TUNING GUIDE
# ═════════════════════════════════════════════════════════════════
st.subheader("🔧 Parameter Tuning Guide")
st.markdown(f"""
| Parameter | Current Value | Effect if Increased |
|---|---|---|
| Lookback Window | {window_size} days | More historical context per sample; slower training |
| Layer 1 Units | {lstm_units_1} | More model capacity; higher risk of overfitting |
| Layer 2 Units | {lstm_units_2} | Same as above |
| Dropout Rate | {dropout_rate} | More regularization; use 0.3–0.4 if overfitting is detected |
| Forecast Horizon | {forecast_days} days | Longer horizon = harder to predict accurately |
| BUY Threshold | {return_threshold * 100:.1f}% | Higher threshold = fewer but stronger BUY signals |
| Model Type | {model_type} | BiLSTM reads both directions; GRU is faster to train |
| Batch Normalization | {"ON" if use_bn else "OFF"} | Stabilizes training; recommended to keep ON |
| Attention Mechanism | {"ON" if use_attention else "OFF"} | Helps model focus on key timesteps; adds complexity |
""")

st.markdown("---")
st.caption(
    "Built with `tf.keras.layers.LSTM` · `GRU` · `Bidirectional` · "
    "Custom Attention Layer · `yfinance` · `Streamlit` · `Plotly` · "
    "PSX / Forex / Commodities support included"
)