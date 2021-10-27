import pandas as pd
import numpy as np
import quandl as quandl
from datetime import date
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

df = quandl.get("BCHAIN/MKPRU", api_key="veu3p-4jnmTcfDVF9SuH").reset_index()   #your quandl api key

btcdata = yf.download(tickers='BTC-USD', period="1d", interval="1m")["Close"]
lastprice = btcdata.iloc[-1]
df.loc[len(df)] = [date.today(), lastprice]

df = df[df["Value"] > 0]
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values(by="Date", inplace=True)
f_date = pd.to_datetime(date(2010, 1, 1))
E_date = pd.to_datetime(date(2009, 1, 3))   #genesis
delta = f_date - E_date
df = df[df.Date > f_date]
df.reset_index(inplace=True)


def normalization(data):
    normalized = (data - data.min()) / (data.max() - data.min())
    return normalized


def ossValue(days):
    X = np.array(np.log10(df.ind[:days])).reshape(-1, 1)
    y = np.array(np.log10(df.Value[:days]))
    reg = LinearRegression().fit(X, y)
    values = reg.predict(X)
    return values[-1]


def normalizationhalving(Normlist):
    global df
    df1 = df[df["Date"] <= "2012-11-01"]
    df2 = df[(df["Date"] > "2012-11-01") & (df["Date"] <= "2015-07-01")]
    df3 = df[(df["Date"] > "2015-07-01") & (df["Date"] <= "2019-06-01")]
    df4 = df[df["Date"] > "2019-06-01"]
    for item in Normlist:
        df1[item].update(normalization(df1[item]))
        df2[item].update(normalization(df2[item]))
        df3[item].update(normalization(df3[item]))
        df4[item].update(normalization(df4[item]))
    df = pd.concat([df1, df2, df3, df4])
    return df

df["Mayer"] = df["Value"] / df["Value"].rolling(200).mean()   #mayer multiple

df["btcIssuance"] = 7200 / 2 ** (np.floor(df["index"] / 1458))
df["usdIssuance"] = df["btcIssuance"] * df["Value"]
df["MAusdIssuance"] = df["usdIssuance"].rolling(window=365).mean()
df["PuellMultiple"] = df["usdIssuance"] / df["MAusdIssuance"]   #puell multiple

df["Price/52w"] = df["Value"] / df["Value"].ewm(span=365).mean()   #price/52w ma

df["Return%"] = df["Value"].pct_change() * 100
df["Sharpe"] = (df["Return%"].rolling(365).mean() - 1) / df["Return%"].rolling(365).std()   #sharpe

dfs = df[df["Return%"] < 0]
df["Sortino"] = (df["Return%"].rolling(365).mean() - 1) / dfs["Return%"].rolling(365).std()   #sortino

df["ind"] = [x + delta.days for x in range(len(df))]
df["PowerLaw"] = np.log10(df.Value) - [ossValue(x + 1) for x in range(len(df))]   #power law

df.update(normalizationhalving(["PuellMultiple", "Price/52w", "PowerLaw", "Sharpe", "Sortino","Mayer"]))
df["avg"] = df[["PuellMultiple","Price/52w", "PowerLaw","Sharpe"]].mean(axis=1)   #avg

fig = make_subplots(specs=[[{"secondary_y": True}]])
xaxis = df.Date

fig.add_trace(go.Scatter(x=xaxis, y=df.Value, name="Price", line=dict(color="gold")), secondary_y=False)
#fig.add_trace(go.Scatter(x=xaxis, y=df["PuellMultiple"], name="Puell Multiple", line=dict(color="white")), secondary_y=True)   #puell multiple
#fig.add_trace(go.Scatter(x=xaxis, y=df["Price/52w"], name="Risk_MA", line=dict(color="white")), secondary_y=True)   #price/52w ma
#fig.add_trace(go.Scatter(x=xaxis, y=df["PowerLaw"], name="power law", line=dict(color="white")), secondary_y=True)   #power law
#fig.add_trace(go.Scatter(x=xaxis, y=df["Sharpe"], name="Sharpe", line=dict(color="white")), secondary_y=True)   #sharpe ratio
#fig.add_trace(go.Scatter(x=xaxis, y=df["Sortino"], name="Sortino", line=dict(color="white")), secondary_y=True)   #sortino ratio
#fig.add_trace(go.Scatter(x=xaxis, y=df["Mayer"], name="Mayer", mode="lines", line=dict(color="white")), secondary_y=True)   #mayer multiple
fig.add_trace(go.Scatter(x=xaxis, y=df["avg"], name="Risk", mode="lines", line=dict(color="white")), secondary_y=True)

fig.add_hrect(y0=0.2, y1=0, line_width=0, fillcolor="green", opacity=0.5, secondary_y=True)
fig.add_hrect(y0=0.3, y1=0.2, line_width=0, fillcolor="green", opacity=0.4, secondary_y=True)
fig.add_hrect(y0=0.4, y1=0.3, line_width=0, fillcolor="green", opacity=0.3, secondary_y=True)
fig.add_hrect(y0=0.5, y1=0.4, line_width=0, fillcolor="green", opacity=0.2, secondary_y=True)
fig.add_hrect(y0=0.6, y1=0.7, line_width=0, fillcolor="red", opacity=0.3, secondary_y=True)
fig.add_hrect(y0=0.7, y1=0.8, line_width=0, fillcolor="red", opacity=0.4, secondary_y=True)
fig.add_hrect(y0=0.8, y1=0.9, line_width=0, fillcolor="red", opacity=0.5, secondary_y=True)
fig.add_hrect(y0=0.9, y1=1.0, line_width=0, fillcolor="red", opacity=0.6, secondary_y=True)

AnnotationText = f"BTC price: {round(df['Value'].iloc[-1])} - Risk: {round(df['avg'].iloc[-1],2)}"

fig.update_layout(xaxis_title='Date', yaxis_title='Price',
                  yaxis2_title='Risk',
                  yaxis1=dict(type='log', showgrid=False),
                  yaxis2=dict(showgrid=True, tickmode='linear', tick0=0.0, dtick=0.1),
                  template="plotly_dark",
                  title={'text': AnnotationText, 'y': 0.9, 'x': 0.5})

fig.show()
