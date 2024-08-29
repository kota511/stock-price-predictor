import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from transformers import pipeline
from tqdm import tqdm
from dotenv import load_dotenv
import praw
import os

# Set parallelism environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
load_dotenv()

# Setup Reddit API
reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent='Sentiment Analysis by u/ka511'
)

def load_stock_data(ticker="AAPL", start_date="1990-01-01"):
    stock_data = yf.Ticker(ticker).history(period="max")
    stock_data = stock_data.loc[start_date:].drop(columns=["Dividends", "Stock Splits"])
    stock_data["Tomorrow"] = stock_data["Close"].shift(-1)
    stock_data["Target"] = (stock_data["Tomorrow"] > stock_data["Close"]).astype(int)
    return stock_data

def plot_candlestick(df, title="Candlestick Chart"):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all")
                ]
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    fig.show()

def collect_reddit_posts(stock, subreddit='stocks', limit=1000, sort='relevance'):
    posts = []
    for submission in reddit.subreddit(subreddit).search(stock, limit=limit, sort=sort):
        posts.append([submission.created_utc, submission.id, submission.title + " " + submission.selftext])
    reddit_df = pd.DataFrame(posts, columns=["Datetime", "Post Id", "Text"])
    reddit_df["Datetime"] = pd.to_datetime(reddit_df["Datetime"], unit='s')
    return reddit_df

def perform_sentiment_analysis(reddit_df, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
    sentiment_task = pipeline("sentiment-analysis", model=model_name, framework="pt", device=0)
    
    def analyze_text(text):
        sentiments = []
        for i in range(0, len(text), 512):
            sentiments.append(sentiment_task(text[i:i + 512])[0])
        return sentiments

    sent_results = {}
    for _, d in tqdm(reddit_df.iterrows(), total=len(reddit_df)):
        sentiments = analyze_text(d["Text"])
        avg_score = sum([s['score'] for s in sentiments]) / len(sentiments)
        label = max(set([s['label'] for s in sentiments]), key=[s['label'] for s in sentiments].count)
        sent_results[d["Post Id"]] = {"label": label, "score": avg_score}
    
    sent_df = pd.DataFrame.from_dict(sent_results, orient='index')
    sent_df = sent_df.merge(reddit_df.set_index("Post Id"), left_index=True, right_index=True)
    sent_df["score_"] = sent_df["score"]
    sent_df.loc[sent_df["label"] == "negative", "score_"] *= -1
    sent_df.loc[sent_df["label"] == "neutral", "score_"] = 0
    sent_df["Date"] = sent_df["Datetime"].dt.date
    sent_daily = sent_df.groupby("Date")["score_"].mean()

    return sent_daily, sent_df

def plot_sentiment_analysis(sent_df):
    sentiment_counts = sent_df['label'].value_counts()
    color_map = {'positive': 'green', 'negative': 'red', 'neutral': 'blue'}
    colors = [color_map[label] for label in sentiment_counts.index]

    fig = go.Figure(data=[go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values, marker=dict(colors=colors))])
    fig.update_layout(title="Sentiment Distribution")
    fig.show()

    fig = px.scatter(
        sent_df,
        x='Date',
        y='score_',
        color='label',
        color_discrete_map=color_map,
        title="Sentiment Scores Over Time"
    )
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all")
                ]
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    fig.show()

def create_predictors_with_sentiment(df, horizons):
    new_predictors = []
    for horizon in horizons:
        rolling_averages = df.rolling(horizon).mean()
        df[f"Close_Ratio_{horizon}"] = df["Close"] / rolling_averages["Close"]
        df[f"Trend_{horizon}"] = df["Target"].rolling(horizon).sum().shift(1)
        new_predictors += [f"Close_Ratio_{horizon}", f"Trend_{horizon}"]
    return df.dropna(), new_predictors

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:i+step].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds = (preds >= 0.6).astype(int)
    return pd.concat([test["Target"], pd.Series(preds, index=test.index, name="Predictions")], axis=1)

def plot_predictions_vs_actual(predictions, start_date=None, end_date=None):
    if start_date and end_date:
        predictions = predictions[(predictions.index >= start_date) & (predictions.index <= end_date)]
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=predictions.index, 
        y=predictions["Target"], 
        mode='lines', 
        name='Actual Target',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=predictions.index, 
        y=predictions["Predictions"], 
        mode='lines', 
        name='Predictions',
        line=dict(color='orange')
    ))

    fig.update_layout(
        title="Predictions vs Actual",
        xaxis_title="Date",
        yaxis_title="Target / Predictions",
        xaxis=dict(
            tickformat="%Y-%m",
            rangeslider=dict(visible=True)
        ),
    )
    
    fig.show()

# Load Apple stock data and plot candlestick chart
apple_stock = load_stock_data()
plot_candlestick(apple_stock, "Apple Inc. Candlestick Chart")

reddit_df = collect_reddit_posts("AAPL", "stocks")
sent_daily, sent_df = perform_sentiment_analysis(reddit_df)
plot_sentiment_analysis(sent_df)

# Apply moving average to smooth the sentiment data
sent_daily_smoothed = sent_daily.rolling(window=30, min_periods=1).mean()

# Reset index to ensure compatibility and avoid timezone issues
apple_stock = apple_stock.reset_index().set_index("Date")
sent_daily_smoothed.index = pd.to_datetime(sent_daily_smoothed.index)

# Ensure both indexes are timezone-naive before merging
apple_stock.index = apple_stock.index.tz_localize(None)
sent_daily_smoothed.index = sent_daily_smoothed.index.tz_localize(None)

# Merge sentiment data with stock data
sent_and_stock = sent_daily_smoothed.to_frame("sentiment").merge(apple_stock, left_index=True, right_index=True)

# Plot combined sentiment and closing price using Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=sent_and_stock.index, 
    y=sent_and_stock["sentiment"], 
    mode='lines', 
    name='Sentiment (Smoothed)',
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    x=sent_and_stock.index, 
    y=sent_and_stock["Close"], 
    mode='lines', 
    name='Closing Price',
    line=dict(color='orange'),
    yaxis="y2"
))

fig.update_layout(
    title="Sentiment and Closing Price Over Time",
    xaxis_title="Date",
    yaxis_title="Sentiment (Smoothed)",
    yaxis2=dict(
        title="Closing Price",
        overlaying="y",
        side="right"
    ),
    xaxis=dict(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all")
            ]
        ),
        rangeslider=dict(visible=True),
        type="date"
    )
)

fig.show()

# Add the smoothed sentiment data as a predictor
apple_stock["Sentiment"] = sent_daily_smoothed.reindex(apple_stock.index, method='ffill').fillna(0)
horizons = [2, 5, 60, 250, 1000]
apple_stock, new_predictors = create_predictors_with_sentiment(apple_stock, horizons)
new_predictors.append("Sentiment")

# Train the model and backtest
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
predictions = backtest(apple_stock, model, new_predictors)

# Plot predictions vs actual
plot_predictions_vs_actual(predictions, start_date="2024-01-01", end_date="2024-03-01")
