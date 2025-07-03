import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import ta
from textblob import TextBlob
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Intraday Trading Strategy - Indian Market",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f2f6, #e8ebf0);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #1f77b4;
    }
    .success-signal {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .danger-signal {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .warning-signal {
        color: #ffc107;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .signal-card {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 3px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Indian stock symbols with more stocks
INDIAN_STOCKS = {
    'Bharat Electronics':'BEL.NS',
    'Indian Railway ':'IRFC.NS',
    'National Petroleum':'NTPC.NS',
    "Indian Oil":'IOC.NS',
    'Reliance Industries': 'RELIANCE.NS',
    'Tata Consultancy Services': 'TCS.NS',
    'HDFC Bank': 'HDFCBANK.NS',
    'Infosys': 'INFY.NS',
    'ICICI Bank': 'ICICIBANK.NS',
    'Hindustan Unilever': 'HINDUNILVR.NS',
    'State Bank of India': 'SBIN.NS',
    'ITC': 'ITC.NS',
    'Bharti Airtel': 'BHARTIARTL.NS',
    'Kotak Mahindra Bank': 'KOTAKBANK.NS',
    'Larsen & Toubro': 'LT.NS',
    'Asian Paints': 'ASIANPAINT.NS',
    'Axis Bank': 'AXISBANK.NS',
    'Maruti Suzuki': 'MARUTI.NS',
    'Bajaj Finance': 'BAJFINANCE.NS',
    'Tata Steel': 'TATASTEEL.NS',
    'Wipro': 'WIPRO.NS',
    'UltraTech Cement': 'ULTRACEMCO.NS',
    'Nestle India': 'NESTLEIND.NS',
    'HCL Technologies': 'HCLTECH.NS',
    'Tech Mahindra': 'TECHM.NS',
    'Sun Pharma': 'SUNPHARMA.NS',
    'Titan Company': 'TITAN.NS',
    'Power Grid Corporation': 'POWERGRID.NS',
    'Bajaj Auto': 'BAJAJ-AUTO.NS',
    'JSW Steel': 'JSWSTEEL.NS',
    'Tata Motors': 'TATAMOTORS.NS',
    'Adani Enterprises': 'ADANIENT.NS',
    'ONGC': 'ONGC.NS',
    'Coal India': 'COALINDIA.NS'
}

# Nifty 50 and Bank Nifty indices
INDICES = {
    'Nifty 50': '^NSEI',
    'Bank Nifty': '^NSEBANK',
    'Nifty IT': '^CNXIT',
    'Nifty Auto': '^CNXAUTO',
    'Nifty Pharma': '^CNXPHARMA'
}

class IntradayTradingSystem:
    def __init__(self):
        self.alpha_vantage_key = "demo"  # Free tier key
        
    def fetch_stock_data(self, symbol, period='1d', interval='5m'):
        """Fetch intraday stock data using Yahoo Finance with error handling"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                st.warning(f"No data available for {symbol}. Market might be closed.")
                return None
                
            return data
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def calculate_technical_indicators(self, data):
        """Calculate enhanced technical indicators"""
        if data is None or len(data) < 50:
            st.warning("Insufficient data for technical analysis. Need at least 50 data points.")
            return None
        
        try:
            # Moving Averages
            data['EMA_9'] = ta.trend.ema_indicator(data['Close'], window=9)
            data['EMA_21'] = ta.trend.ema_indicator(data['Close'], window=21)
            data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
            data['SMA_200'] = ta.trend.sma_indicator(data['Close'], window=min(200, len(data)))
            
            # RSI
            data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
            
            # MACD
            macd = ta.trend.MACD(data['Close'])
            data['MACD'] = macd.macd()
            data['MACD_Signal'] = macd.macd_signal()
            data['MACD_Histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(data['Close'])
            data['BB_Upper'] = bollinger.bollinger_hband()
            data['BB_Middle'] = bollinger.bollinger_mavg()
            data['BB_Lower'] = bollinger.bollinger_lband()
            data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
            
            # Stochastic
            data['Stoch_K'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'])
            data['Stoch_D'] = ta.momentum.stoch_signal(data['High'], data['Low'], data['Close'])
            
            # ADX for trend strength
            data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'])
            data['DI_Plus'] = ta.trend.adx_pos(data['High'], data['Low'], data['Close'])
            data['DI_Minus'] = ta.trend.adx_neg(data['High'], data['Low'], data['Close'])
            
            # Volume indicators
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
            data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
            data['VWAP'] = ta.volume.volume_weighted_average_price(data['High'], data['Low'], data['Close'], data['Volume'])
            
            # Volatility
            data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
            
            # Support and Resistance levels using pivot points
            data['Pivot'] = (data['High'] + data['Low'] + data['Close']) / 3
            data['Support1'] = (2 * data['Pivot']) - data['High']
            data['Resistance1'] = (2 * data['Pivot']) - data['Low']
            data['Support2'] = data['Pivot'] - (data['High'] - data['Low'])
            data['Resistance2'] = data['Pivot'] + (data['High'] - data['Low'])
            
            # Williams %R
            data['Williams_R'] = ta.momentum.williams_r(data['High'], data['Low'], data['Close'])
            
            # Commodity Channel Index
            data['CCI'] = ta.trend.cci(data['High'], data['Low'], data['Close'])
            
            # Momentum
            data['Momentum'] = data['Close'].pct_change(periods=10) * 100
            
            return data
            
        except Exception as e:
            st.error(f"Error calculating technical indicators: {str(e)}")
            return None
    
    def get_news_sentiment(self, symbol):
        """Enhanced news sentiment analysis"""
        try:
            # Remove .NS suffix for API call
            clean_symbol = symbol.replace('.NS', '')
            
            # Try Alpha Vantage first
            url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={clean_symbol}&apikey={self.alpha_vantage_key}'
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'feed' in data and len(data['feed']) > 0:
                    # Calculate average sentiment
                    sentiments = []
                    for article in data['feed'][:10]:  # Top 10 articles
                        if 'overall_sentiment_score' in article:
                            sentiments.append(float(article['overall_sentiment_score']))
                    
                    if sentiments:
                        avg_sentiment = np.mean(sentiments)
                        return avg_sentiment, len(sentiments)
            
            # Fallback: Simple sentiment analysis on company name
            company_name = [k for k, v in INDIAN_STOCKS.items() if v == symbol][0]
            
            return self.simple_sentiment_analysis(company_name), 1
            
        except Exception as e:
            st.info(f"Using basic sentiment analysis due to API limitations")
            return 0, 0
    
    def simple_sentiment_analysis(self, text):
        """Simple sentiment analysis using TextBlob"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0
    
    def generate_signals(self, data, sentiment_score=0):
        """Enhanced signal generation with more sophisticated logic"""
        if data is None or len(data) < 50:
            return None
        
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        signals = {
            'overall_signal': 'HOLD',
            'confidence': 0,
            'entry_price': latest['Close'],
            'stop_loss': 0,
            'target': 0,
            'signals_breakdown': {},
            'risk_reward_ratio': 0,
            'trend_strength': 'WEAK'
        }
        
        signal_points = 0
        max_points = 10  # Increased for more indicators
        
        # RSI Signal (Enhanced)
        if latest['RSI'] < 25:
            signals['signals_breakdown']['RSI'] = 'EXTREMELY OVERSOLD - STRONG BUY'
            signal_points += 2
        elif latest['RSI'] < 30:
            signals['signals_breakdown']['RSI'] = 'OVERSOLD - BUY'
            signal_points += 1
        elif latest['RSI'] > 75:
            signals['signals_breakdown']['RSI'] = 'EXTREMELY OVERBOUGHT - STRONG SELL'
            signal_points -= 2
        elif latest['RSI'] > 70:
            signals['signals_breakdown']['RSI'] = 'OVERBOUGHT - SELL'
            signal_points -= 1
        else:
            signals['signals_breakdown']['RSI'] = f'NEUTRAL ({latest["RSI"]:.1f})'
        
        # MACD Signal
        if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
            signals['signals_breakdown']['MACD'] = 'BULLISH CROSSOVER - BUY'
            signal_points += 1
        elif latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
            signals['signals_breakdown']['MACD'] = 'BEARISH CROSSOVER - SELL'
            signal_points -= 1
        elif latest['MACD'] > latest['MACD_Signal']:
            signals['signals_breakdown']['MACD'] = 'ABOVE SIGNAL LINE - BULLISH'
            signal_points += 0.5
        elif latest['MACD'] < latest['MACD_Signal']:
            signals['signals_breakdown']['MACD'] = 'BELOW SIGNAL LINE - BEARISH'
            signal_points -= 0.5
        else:
            signals['signals_breakdown']['MACD'] = 'NEUTRAL'
        
        # Bollinger Bands Signal
        bb_position = (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])
        if bb_position <= 0.1:
            signals['signals_breakdown']['Bollinger'] = 'NEAR LOWER BAND - BUY'
            signal_points += 1
        elif bb_position >= 0.9:
            signals['signals_breakdown']['Bollinger'] = 'NEAR UPPER BAND - SELL'
            signal_points -= 1
        else:
            signals['signals_breakdown']['Bollinger'] = f'WITHIN BANDS ({bb_position:.1%})'
        
        # EMA Signal
        if latest['EMA_9'] > latest['EMA_21'] and prev['EMA_9'] <= prev['EMA_21']:
            signals['signals_breakdown']['EMA'] = 'GOLDEN CROSS - BUY'
            signal_points += 1.5
        elif latest['EMA_9'] < latest['EMA_21'] and prev['EMA_9'] >= prev['EMA_21']:
            signals['signals_breakdown']['EMA'] = 'DEATH CROSS - SELL'
            signal_points -= 1.5
        elif latest['EMA_9'] > latest['EMA_21']:
            signals['signals_breakdown']['EMA'] = 'BULLISH ALIGNMENT'
            signal_points += 0.5
        else:
            signals['signals_breakdown']['EMA'] = 'BEARISH ALIGNMENT'
            signal_points -= 0.5
        
        # Stochastic Signal
        if latest['Stoch_K'] < 20 and latest['Stoch_D'] < 20 and latest['Stoch_K'] > latest['Stoch_D']:
            signals['signals_breakdown']['Stochastic'] = 'OVERSOLD BULLISH CROSSOVER - BUY'
            signal_points += 1.5
        elif latest['Stoch_K'] > 80 and latest['Stoch_D'] > 80 and latest['Stoch_K'] < latest['Stoch_D']:
            signals['signals_breakdown']['Stochastic'] = 'OVERBOUGHT BEARISH CROSSOVER - SELL'
            signal_points -= 1.5
        elif latest['Stoch_K'] < 20:
            signals['signals_breakdown']['Stochastic'] = 'OVERSOLD - POTENTIAL BUY'
            signal_points += 0.5
        elif latest['Stoch_K'] > 80:
            signals['signals_breakdown']['Stochastic'] = 'OVERBOUGHT - POTENTIAL SELL'
            signal_points -= 0.5
        else:
            signals['signals_breakdown']['Stochastic'] = f'NEUTRAL ({latest["Stoch_K"]:.1f})'
        
        # ADX Trend Strength
        if latest['ADX'] > 40:
            signals['signals_breakdown']['ADX'] = f'VERY STRONG TREND ({latest["ADX"]:.1f})'
            signals['trend_strength'] = 'VERY STRONG'
            signal_points += 1 if signal_points > 0 else -1
        elif latest['ADX'] > 25:
            signals['signals_breakdown']['ADX'] = f'STRONG TREND ({latest["ADX"]:.1f})'
            signals['trend_strength'] = 'STRONG'
            signal_points += 0.5 if signal_points > 0 else -0.5
        else:
            signals['signals_breakdown']['ADX'] = f'WEAK TREND ({latest["ADX"]:.1f})'
            signals['trend_strength'] = 'WEAK'
        
        # Volume Signal
        avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
        volume_ratio = latest['Volume'] / avg_volume
        if volume_ratio > 2:
            signals['signals_breakdown']['Volume'] = f'VERY HIGH VOLUME ({volume_ratio:.1f}x) - STRONG SIGNAL'
            signal_points += 1 if signal_points > 0 else -1
        elif volume_ratio > 1.5:
            signals['signals_breakdown']['Volume'] = f'HIGH VOLUME ({volume_ratio:.1f}x) - CONFIRMATION'
            signal_points += 0.5 if signal_points > 0 else -0.5
        else:
            signals['signals_breakdown']['Volume'] = f'NORMAL VOLUME ({volume_ratio:.1f}x)'
        
        # Williams %R Signal
        if latest['Williams_R'] > -20:
            signals['signals_breakdown']['Williams_R'] = 'OVERBOUGHT - SELL SIGNAL'
            signal_points -= 0.5
        elif latest['Williams_R'] < -80:
            signals['signals_breakdown']['Williams_R'] = 'OVERSOLD - BUY SIGNAL'
            signal_points += 0.5
        else:
            signals['signals_breakdown']['Williams_R'] = 'NEUTRAL'
        
        # Sentiment Signal
        if sentiment_score > 0.2:
            signals['signals_breakdown']['Sentiment'] = 'VERY POSITIVE NEWS - STRONG BULLISH'
            signal_points += 1
        elif sentiment_score > 0.1:
            signals['signals_breakdown']['Sentiment'] = 'POSITIVE NEWS - BULLISH'
            signal_points += 0.5
        elif sentiment_score < -0.2:
            signals['signals_breakdown']['Sentiment'] = 'VERY NEGATIVE NEWS - STRONG BEARISH'
            signal_points -= 1
        elif sentiment_score < -0.1:
            signals['signals_breakdown']['Sentiment'] = 'NEGATIVE NEWS - BEARISH'
            signal_points -= 0.5
        else:
            signals['signals_breakdown']['Sentiment'] = 'NEUTRAL NEWS'
        
        # Overall Signal with enhanced logic
        signal_strength = signal_points / max_points
        
        if signal_points >= 3:
            signals['overall_signal'] = 'STRONG BUY'
            signals['confidence'] = min(95, abs(signal_strength) * 100)
        elif signal_points >= 1.5:
            signals['overall_signal'] = 'BUY'
            signals['confidence'] = min(80, abs(signal_strength) * 100)
        elif signal_points <= -3:
            signals['overall_signal'] = 'STRONG SELL'
            signals['confidence'] = min(95, abs(signal_strength) * 100)
        elif signal_points <= -1.5:
            signals['overall_signal'] = 'SELL'
            signals['confidence'] = min(80, abs(signal_strength) * 100)
        else:
            signals['overall_signal'] = 'HOLD'
            signals['confidence'] = 40
        
        # Calculate Stop Loss and Target using ATR and support/resistance
        atr = latest['ATR']
        if 'BUY' in signals['overall_signal']:
            signals['stop_loss'] = max(latest['Close'] - (2 * atr), latest['Support1'])
            signals['target'] = min(latest['Close'] + (3 * atr), latest['Resistance1'])
        elif 'SELL' in signals['overall_signal']:
            signals['stop_loss'] = min(latest['Close'] + (2 * atr), latest['Resistance1'])
            signals['target'] = max(latest['Close'] - (3 * atr), latest['Support1'])
        
        # Calculate Risk-Reward Ratio
        if signals['stop_loss'] != 0:
            risk = abs(latest['Close'] - signals['stop_loss'])
            reward = abs(signals['target'] - latest['Close'])
            if risk > 0:
                signals['risk_reward_ratio'] = reward / risk
        
        return signals
    
    def create_enhanced_chart(self, data, signals):
        """Create enhanced interactive chart with more indicators"""
        if data is None:
            return None
        
        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=('Price & Indicators', 'MACD', 'RSI & Stochastic', 'Volume', 'Williams %R'),
            row_heights=[0.4, 0.15, 0.15, 0.15, 0.15]
        )
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ), row=1, col=1)
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_Upper'],
            line=dict(color='rgba(255,0,0,0.5)', width=1),
            name='BB Upper'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_Lower'],
            line=dict(color='rgba(255,0,0,0.5)', width=1),
            name='BB Lower',
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)'
        ), row=1, col=1)
        
        # EMAs
        fig.add_trace(go.Scatter(
            x=data.index, y=data['EMA_9'],
            line=dict(color='orange', width=2),
            name='EMA 9'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data['EMA_21'],
            line=dict(color='blue', width=2),
            name='EMA 21'
        ), row=1, col=1)
        
        # VWAP
        fig.add_trace(go.Scatter(
            x=data.index, y=data['VWAP'],
            line=dict(color='purple', width=2, dash='dash'),
            name='VWAP'
        ), row=1, col=1)
        
        # Support and Resistance
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Support1'],
            line=dict(color='green', width=1, dash='dot'),
            name='Support 1'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Resistance1'],
            line=dict(color='red', width=1, dash='dot'),
            name='Resistance 1'
        ), row=1, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(
            x=data.index, y=data['MACD'],
            line=dict(color='blue'),
            name='MACD'
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data['MACD_Signal'],
            line=dict(color='red'),
            name='MACD Signal'
        ), row=2, col=1)
        
        fig.add_trace(go.Bar(
            x=data.index, y=data['MACD_Histogram'],
            name='MACD Histogram'
        ), row=2, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(
            x=data.index, y=data['RSI'],
            line=dict(color='purple'),
            name='RSI'
        ), row=3, col=1)
        
        # Stochastic
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Stoch_K'],
            line=dict(color='orange'),
            name='Stoch %K'
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Stoch_D'],
            line=dict(color='blue'),
            name='Stoch %D'
        ), row=3, col=1)
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=3, col=1)
        
        # Volume
        colors = ['red' if close < open else 'green' for close, open in zip(data['Close'], data['Open'])]
        fig.add_trace(go.Bar(
            x=data.index, y=data['Volume'],
            name='Volume',
            marker_color=colors
        ), row=4, col=1)
        
        # Volume SMA
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Volume_SMA'],
            line=dict(color='blue', width=2),
            name='Volume SMA'
        ), row=4, col=1)
        
        # Williams %R
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Williams_R'],
            line=dict(color='red'),
            name='Williams %R'
        ), row=5, col=1)
        
        fig.add_hline(y=-20, line_dash="dash", line_color="red", row=5, col=1)
        fig.add_hline(y=-80, line_dash="dash", line_color="green", row=5, col=1)
        
        fig.update_layout(
            title='Enhanced Technical Analysis Chart',
            xaxis_rangeslider_visible=False,
            height=900,
            showlegend=True
        )
        
        return fig

# Enhanced Streamlit App
def main():
    st.markdown('<h1 class="main-header">🚀 Advanced Intraday Trading Strategy System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Powered by Advanced Technical Indicators, Sentiment Analysis & Real-time Market Data</p>', unsafe_allow_html=True)
    
    system = IntradayTradingSystem()
    
    # Sidebar
    st.sidebar.markdown("## 📊 Configuration")
    
    # Market selection
    market_type = st.sidebar.selectbox(
        "Select Market Type:",
        ["Individual Stocks", "Market Indices"]
    )
    # Add this after market_type selection:
    if market_type == "Individual Stocks":
        input_method = st.sidebar.radio("Stock Selection Method:", ["From List", "Manual Entry"])
        if input_method == "From List":
            selected_stock_name = st.sidebar.selectbox("Select Stock:", list(INDIAN_STOCKS.keys()))
            selected_symbol = INDIAN_STOCKS[selected_stock_name]
        else:
            custom_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., RELIANCE.NS):", "RELIANCE.NS")
            selected_symbol = custom_symbol
            selected_stock_name = custom_symbol.replace('.NS', '')
    # if market_type == "Individual Stocks":
    #     # Stock selection
    #     selected_stock_name = st.sidebar.selectbox(
    #         "Select Stock:",
    #         list(INDIAN_STOCKS.keys())
    #     )
    #     selected_symbol = INDIAN_STOCKS[selected_stock_name]
    # else:
    #     # Index selection
    #     selected_index_name = st.sidebar.selectbox(
    #         "Select Index:",
    #         list(INDICES.keys())
    #     )
    #     selected_symbol = INDICES[selected_index_name]
    #     selected_stock_name = selected_index_name
    
    # Time frame selection
    time_frame = st.sidebar.selectbox(
        "Select Time Frame:",
        ["1d", "5d", "1mo", "3mo"],
        help="1d = Today, 5d = Last 5 days, 1mo = Last month, 3mo = Last 3 months"
    )
    
    # Interval selection
    interval = st.sidebar.selectbox(
        "Select Interval:",
        ["1m", "2m", "5m", "15m", "30m", "1h", "1d"],
        index=2,
        help="1m = 1 minute, 2m = 2 minutes, 5m = 5 minutes, etc."
    )
    
    # Real-time updates
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30 seconds)", value=False)
    
    # Analysis button
    if st.sidebar.button("🔍 Analyze Stock", type="primary") or auto_refresh:
        if auto_refresh:
            import time
            time.sleep(1)  # Small delay for auto-refresh
            
        with st.spinner("Fetching data and analyzing..."):
            # Fetch data
            data = system.fetch_stock_data(selected_symbol, time_frame, interval)
            
            if data is not None and not data.empty:
                # Calculate indicators
                data_with_indicators = system.calculate_technical_indicators(data)
                
                if data_with_indicators is not None:
                    # Get sentiment (only for stocks, not indices)
                    sentiment_score, sentiment_count = (0, 0)
                    if market_type == "Individual Stocks":
                        sentiment_score, sentiment_count = system.get_news_sentiment(selected_symbol)
                    
                    # Generate signals
                    signals = system.generate_signals(data_with_indicators, sentiment_score)
                    
                    if signals:
                        # Display current time
                        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        # Key metrics row
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            signal_class = "success-signal" if "BUY" in signals['overall_signal'] else "danger-signal" if "SELL" in signals['overall_signal'] else "warning-signal"
                            st.markdown(f'<div class="metric-card"><h3>Overall Signal</h3><p class="{signal_class}">{signals["overall_signal"]}</p></div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f'<div class="metric-card"><h3>Confidence</h3><p style="font-size: 1.2rem; font-weight: bold;">{signals["confidence"]:.1f}%</p></div>', unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f'<div class="metric-card"><h3>Current Price</h3><p style="font-size: 1.2rem; font-weight: bold;">₹{signals["entry_price"]:.2f}</p></div>', unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown(f'<div class="metric-card"><h3>Trend Strength</h3><p style="font-size: 1.1rem; font-weight: bold;">{signals["trend_strength"]}</p></div>', unsafe_allow_html=True)
                        
                        # Price change calculation
                        price_change = ((data_with_indicators['Close'].iloc[-1] - data_with_indicators['Close'].iloc[-2]) / data_with_indicators['Close'].iloc[-2]) * 100
                        price_change_color = "green" if price_change > 0 else "red"
                        
                        # Additional metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(f'<div class="metric-card"><h3>Price Change</h3><p style="color: {price_change_color}; font-size: 1.1rem; font-weight: bold;">{price_change:+.2f}%</p></div>', unsafe_allow_html=True)
                        
                        with col2:
                            if signals['risk_reward_ratio'] > 0:
                                st.markdown(f'<div class="metric-card"><h3>Risk:Reward</h3><p style="font-size: 1.1rem; font-weight: bold;">1:{signals["risk_reward_ratio"]:.2f}</p></div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="metric-card"><h3>Risk:Reward</h3><p style="font-size: 1.1rem;">N/A</p></div>', unsafe_allow_html=True)
                        
                        with col3:
                            latest_volume = data_with_indicators['Volume'].iloc[-1]
                            avg_volume = data_with_indicators['Volume'].rolling(window=20).mean().iloc[-1]
                            volume_ratio = latest_volume / avg_volume
                            st.markdown(f'<div class="metric-card"><h3>Volume Ratio</h3><p style="font-size: 1.1rem; font-weight: bold;">{volume_ratio:.2f}x</p></div>', unsafe_allow_html=True)
                        
                        with col4:
                            volatility = data_with_indicators['ATR'].iloc[-1]
                            st.markdown(f'<div class="metric-card"><h3>Volatility (ATR)</h3><p style="font-size: 1.1rem; font-weight: bold;">₹{volatility:.2f}</p></div>', unsafe_allow_html=True)
                        
                        # Signal breakdown
                        st.markdown("## 📈 Detailed Signal Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 🔍 Technical Indicators")
                            for indicator, signal in signals['signals_breakdown'].items():
                                if indicator not in ['Sentiment']:
                                    if "BUY" in signal:
                                        signal_emoji = "🟢"
                                        signal_style = "color: #28a745;"
                                    elif "SELL" in signal:
                                        signal_emoji = "🔴"
                                        signal_style = "color: #dc3545;"
                                    else:
                                        signal_emoji = "🟡"
                                        signal_style = "color: #ffc107;"
                                    
                                    st.markdown(f'<div class="signal-card">{signal_emoji} <strong>{indicator}</strong>: <span style="{signal_style}">{signal}</span></div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("### 💼 Trade Setup")
                            if signals['stop_loss'] > 0:
                                st.markdown(f'<div class="signal-card"><strong>Entry Price:</strong> ₹{signals["entry_price"]:.2f}</div>', unsafe_allow_html=True)
                                st.markdown(f'<div class="signal-card"><strong>Stop Loss:</strong> ₹{signals["stop_loss"]:.2f}</div>', unsafe_allow_html=True)
                                st.markdown(f'<div class="signal-card"><strong>Target:</strong> ₹{signals["target"]:.2f}</div>', unsafe_allow_html=True)
                                
                                risk = abs(signals['entry_price'] - signals['stop_loss'])
                                reward = abs(signals['target'] - signals['entry_price'])
                                risk_percent = (risk / signals['entry_price']) * 100
                                reward_percent = (reward / signals['entry_price']) * 100
                                
                                st.markdown(f'<div class="signal-card"><strong>Risk:</strong> {risk_percent:.2f}% (₹{risk:.2f})</div>', unsafe_allow_html=True)
                                st.markdown(f'<div class="signal-card"><strong>Reward:</strong> {reward_percent:.2f}% (₹{reward:.2f})</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="signal-card"><em>No active trade setup for HOLD signal</em></div>', unsafe_allow_html=True)
                        
                        # Sentiment analysis (only for stocks)
                        if market_type == "Individual Stocks":
                            st.markdown("## 📰 Market Sentiment")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                sentiment_text = "Very Positive 😊😊" if sentiment_score > 0.2 else "Positive 😊" if sentiment_score > 0.1 else "Very Negative 😞😞" if sentiment_score < -0.2 else "Negative 😞" if sentiment_score < -0.1 else "Neutral 😐"
                                sentiment_color = "#28a745" if sentiment_score > 0.1 else "#dc3545" if sentiment_score < -0.1 else "#ffc107"
                                st.markdown(f'<div class="signal-card"><strong>News Sentiment:</strong> <span style="color: {sentiment_color};">{sentiment_text}</span></div>', unsafe_allow_html=True)
                                st.markdown(f'<div class="signal-card"><strong>Sentiment Score:</strong> {sentiment_score:.3f}</div>', unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f'<div class="signal-card"><strong>Articles Analyzed:</strong> {sentiment_count}</div>', unsafe_allow_html=True)
                                if 'Sentiment' in signals['signals_breakdown']:
                                    sentiment_signal = signals['signals_breakdown']['Sentiment']
                                    st.markdown(f'<div class="signal-card"><strong>Sentiment Signal:</strong> {sentiment_signal}</div>', unsafe_allow_html=True)
                        
                        # Chart
                        st.markdown("## 📊 Advanced Technical Analysis Chart")
                        chart = system.create_enhanced_chart(data_with_indicators, signals)
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
                        
                        # Current market data
                        st.markdown("## 📋 Current Market Snapshot")
                        latest_data = data_with_indicators.iloc[-1]
                        
                        # Create tabs for different indicator groups
                        tab1, tab2, tab3, tab4 = st.tabs(["📈 Trend Indicators", "🔄 Momentum Indicators", "📊 Volume Analysis", "🎯 Support/Resistance"])
                        
                        with tab1:
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("EMA 9", f"₹{latest_data['EMA_9']:.2f}")
                                st.metric("EMA 21", f"₹{latest_data['EMA_21']:.2f}")
                            
                            with col2:
                                st.metric("SMA 50", f"₹{latest_data['SMA_50']:.2f}")
                                st.metric("VWAP", f"₹{latest_data['VWAP']:.2f}")
                            
                            with col3:
                                st.metric("ADX", f"{latest_data['ADX']:.2f}")
                                st.metric("DI+", f"{latest_data['DI_Plus']:.2f}")
                            
                            with col4:
                                st.metric("DI-", f"{latest_data['DI_Minus']:.2f}")
                                st.metric("MACD", f"{latest_data['MACD']:.4f}")
                        
                        with tab2:
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("RSI", f"{latest_data['RSI']:.2f}")
                                st.metric("Stoch %K", f"{latest_data['Stoch_K']:.2f}")
                            
                            with col2:
                                st.metric("Stoch %D", f"{latest_data['Stoch_D']:.2f}")
                                st.metric("Williams %R", f"{latest_data['Williams_R']:.2f}")
                            
                            with col3:
                                st.metric("CCI", f"{latest_data['CCI']:.2f}")
                                st.metric("Momentum", f"{latest_data['Momentum']:.2f}%")
                            
                            with col4:
                                bb_position = ((latest_data['Close'] - latest_data['BB_Lower']) / (latest_data['BB_Upper'] - latest_data['BB_Lower']) * 100)
                                st.metric("BB Position", f"{bb_position:.1f}%")
                                st.metric("BB Width", f"₹{latest_data['BB_Width']:.2f}")
                        
                        with tab3:
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Current Volume", f"{latest_data['Volume']:,}")
                                st.metric("Avg Volume (20)", f"{latest_data['Volume_SMA']:,.0f}")
                            
                            with col2:
                                volume_ratio = latest_data['Volume'] / latest_data['Volume_SMA']
                                st.metric("Volume Ratio", f"{volume_ratio:.2f}x")
                                st.metric("OBV", f"{latest_data['OBV']:,.0f}")
                            
                            with col3:
                                st.metric("ATR", f"₹{latest_data['ATR']:.2f}")
                                atr_percent = (latest_data['ATR'] / latest_data['Close']) * 100
                                st.metric("ATR %", f"{atr_percent:.2f}%")
                            
                            with col4:
                                high_low_range = latest_data['High'] - latest_data['Low']
                                st.metric("Day Range", f"₹{high_low_range:.2f}")
                                range_percent = (high_low_range / latest_data['Close']) * 100
                                st.metric("Range %", f"{range_percent:.2f}%")
                        
                        with tab4:
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Support 1", f"₹{latest_data['Support1']:.2f}")
                                st.metric("Support 2", f"₹{latest_data['Support2']:.2f}")
                            
                            with col2:
                                st.metric("Resistance 1", f"₹{latest_data['Resistance1']:.2f}")
                                st.metric("Resistance 2", f"₹{latest_data['Resistance2']:.2f}")
                            
                            with col3:
                                st.metric("Pivot Point", f"₹{latest_data['Pivot']:.2f}")
                                pivot_distance = ((latest_data['Close'] - latest_data['Pivot']) / latest_data['Pivot']) * 100
                                st.metric("Distance from Pivot", f"{pivot_distance:+.2f}%")
                            
                            with col4:
                                support_distance = ((latest_data['Close'] - latest_data['Support1']) / latest_data['Support1']) * 100
                                resistance_distance = ((latest_data['Resistance1'] - latest_data['Close']) / latest_data['Close']) * 100
                                st.metric("To Support", f"{support_distance:.2f}%")
                                st.metric("To Resistance", f"{resistance_distance:.2f}%")
                        
                        # Trading recommendations
                        st.markdown("## 🎯 Trading Recommendations")
                        
                        if signals['overall_signal'] in ['STRONG BUY', 'BUY']:
                            st.success(f"""
                            **📈 {signals['overall_signal']} Signal Detected!**
                            
                            **Recommended Action:** Consider a long position
                            **Entry:** ₹{signals['entry_price']:.2f}
                            **Stop Loss:** ₹{signals['stop_loss']:.2f}
                            **Target:** ₹{signals['target']:.2f}
                            **Risk per share:** ₹{abs(signals['entry_price'] - signals['stop_loss']):.2f}
                            **Reward per share:** ₹{abs(signals['target'] - signals['entry_price']):.2f}
                            
                            **Position Size Calculation:**
                            - Risk 1% of capital: If you have ₹1,00,000, risk ₹1,000
                            - Shares to buy: ₹1,000 ÷ ₹{abs(signals['entry_price'] - signals['stop_loss']):.2f} = {int(1000 / abs(signals['entry_price'] - signals['stop_loss']))} shares
                            """)
                        
                        elif signals['overall_signal'] in ['STRONG SELL', 'SELL']:
                            st.error(f"""
                            **📉 {signals['overall_signal']} Signal Detected!**
                            
                            **Recommended Action:** Consider a short position or avoid buying
                            **Entry:** ₹{signals['entry_price']:.2f}
                            **Stop Loss:** ₹{signals['stop_loss']:.2f}
                            **Target:** ₹{signals['target']:.2f}
                            **Risk per share:** ₹{abs(signals['entry_price'] - signals['stop_loss']):.2f}
                            **Reward per share:** ₹{abs(signals['target'] - signals['entry_price']):.2f}
                            
                            **Note:** Short selling requires margin account and has additional risks.
                            """)
                        
                        else:
                            st.warning(f"""
                            **⏸️ HOLD Signal**
                            
                            **Recommended Action:** Wait for clearer signals
                            **Current Price:** ₹{signals['entry_price']:.2f}
                            **Market Condition:** {signals['trend_strength']} trend
                            
                            **Suggestion:** Monitor for breakout above resistance (₹{latest_data['Resistance1']:.2f}) or breakdown below support (₹{latest_data['Support1']:.2f})
                            """)
                        
                        # Risk management tips
                        st.markdown("## ⚠️ Risk Management Guidelines")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("""
                            **Position Sizing:**
                            - Never risk more than 1-2% of your capital on a single trade
                            - Use the ATR to calculate position size
                            - Consider market volatility before entering
                            
                            **Entry Rules:**
                            - Wait for multiple confirmations
                            - Enter on pullbacks in trending markets
                            - Avoid trading during high-impact news events
                            """)
                        
                        with col2:
                            st.markdown("""
                            **Exit Rules:**
                            - Always set stop-loss before entering
                            - Trail stop-loss in profitable trades
                            - Take partial profits at resistance levels
                            
                            **Market Conditions:**
                            - Strong trends: Follow the trend
                            - Weak trends: Look for reversals
                            - High volatility: Reduce position size
                            """)
                        
                        # Auto-refresh for real-time updates
                        if auto_refresh:
                            st.rerun()
                    
                    else:
                        st.error("Unable to generate trading signals. Please try again.")
                else:
                    st.error("Error calculating technical indicators. Please check the data.")
            else:
                st.error("Unable to fetch market data. Please check your internet connection or try again later.")
        
        # Market status
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 🕒 Market Status")
        current_time = datetime.now()
        market_open = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
        
        if market_open <= current_time <= market_close and current_time.weekday() < 5:
            st.sidebar.success("🟢 Market is OPEN")
        else:
            st.sidebar.error("🔴 Market is CLOSED")
        
        st.sidebar.markdown(f"**Current Time:** {current_time.strftime('%H:%M:%S')}")
        st.sidebar.markdown(f"**Market Hours:** 09:15 - 15:30 (Mon-Fri)")
    
    # Instructions
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📖 How to Use")
    st.sidebar.markdown("""
    1. **Select market type** (Stocks or Indices)
    2. **Choose your instrument** from the dropdown
    3. **Set time frame** (1d for intraday, 5d for short-term)
    4. **Pick interval** (1m-5m for scalping, 15m-1h for swing)
    5. **Click Analyze** to get comprehensive signals
    6. **Review all indicators** and market conditions
    7. **Follow risk management** guidelines
    8. **Enable auto-refresh** for real-time updates
    """)
    
    st.sidebar.markdown("## 🎯 Enhanced Features")
    st.sidebar.markdown("""
    - **15+ Technical Indicators**: RSI, MACD, Bollinger Bands, EMA, Stochastic, ADX, Williams %R, CCI, etc.
    - **Advanced Chart**: 5-panel layout with all indicators
    - **Sentiment Analysis**: News-based market sentiment
    - **Volume Analysis**: Above average volume detection
    - **Support/Resistance**: Dynamic pivot points
    - **Risk Management**: Position sizing calculator
    - **Real-time Updates**: Auto-refresh capability
    - **Market Status**: Live market hours tracking
    - **Enhanced Visuals**: Professional styling and layout
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("## ⚠️ Important Disclaimer")
    st.warning("""
    **This application is for educational and informational purposes only.**
    
    - Trading involves substantial risk of loss
    - Past performance does not guarantee future results
    - Always do your own research and analysis
    - Consult with a qualified financial advisor before making investment decisions
    - The creators are not responsible for any trading losses
    - Use proper risk management and position sizing
    - Never invest more than you can afford to lose
    """)
    
    st.markdown("---")
    st.markdown("*Developed for Indian Stock Market Analysis • Data provided by Yahoo Finance*")

if __name__ == "__main__":
    main()