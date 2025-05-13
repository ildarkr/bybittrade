
# ОБЪЕДИНЁННЫЙ БОТ: автотрейд + прогноз + цены входа/выхода
import requests, time, hmac, hashlib, json, telebot, threading
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

API_KEY = 'VGRSv37ainnba2vaIV'.strip()
API_SECRET = 'r0I7KpNibNTLNgHuSy6NzVoGS1PuJKWLPtxp'.strip()
BOT_TOKEN = '7508493916:AAEzPU2ev7WPJyd4ACtyC8a5m8iC6cNv9JU'
CHAT_ID = '886410050'

bot = telebot.TeleBot(BOT_TOKEN)
SYMBOL = "ETHUSDT"
CAPITAL = 25
LEVERAGE = 25
TP_PCT = 1.5
SL_PCT = 1.0

def place_order(side, qty, tp, sl):
    url = "https://api.bybit.com/v5/order/create"
    req_time = str(int(time.time() * 1000))

    body_dict = {
        "category": "linear",
        "symbol": SYMBOL,
        "side": side,
        "orderType": "Market",
        "qty": qty,
        "timeInForce": "GoodTillCancel",
        "isLeverage": True,
        "positionIdx": 1,
        "takeProfit": str(tp),
        "stopLoss": str(sl)
    }

    body = json.dumps(body_dict, separators=(",", ":"))
    signature_payload = f"{req_time}{API_KEY}{body}"
    signature = hmac.new(API_SECRET.encode(), signature_payload.encode(), hashlib.sha256).hexdigest()

    headers = {
        "X-BYBIT-API-KEY": API_KEY,
        "X-BYBIT-SIGN": signature,
        "X-BYBIT-TIMESTAMP": req_time,
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, data=body)
    return response.json()  
def get_price():
    url = "https://api.bybit.com/v5/market/tickers?category=linear&symbol=ETHUSDT"
    try:
        r = requests.get(url)
        return float(r.json()["result"]["list"][0]["lastPrice"])
    except Exception as e:
        print("Ошибка получения цены:", e)
        return None

def get_klines(interval="5", limit=1000):
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={SYMBOL}&interval={interval}&limit={limit}"
    try:
        r = requests.get(url)
        data = r.json()
        df = pd.DataFrame(data["result"]["list"])
        df.columns = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df[::-1].reset_index(drop=True)
    except:
        return None

def compute_rsi(series, window):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_indicators(df):
    df["MA"] = df["close"].rolling(window=10).mean()
    df["EMA"] = df["close"].ewm(span=10).mean()
    df["RSI"] = compute_rsi(df["close"], 14)
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["future_close"] = df["close"].shift(-1)
    df["target"] = (df["future_close"] > df["close"]).astype(int)
    df.dropna(inplace=True)
    return df

def train_model(df):
    X = df[["MA", "EMA", "RSI", "MACD"]]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, acc

def predict(df, model):
    return model.predict(df[["MA", "EMA", "RSI", "MACD"]].iloc[-1:])[0]

def forecast_trade():
    df = get_klines()
    if df is None:
        bot.send_message(CHAT_ID, "Ошибка: не удалось получить данные с графика.")
        return

    df = calculate_indicators(df)
    model, acc = train_model(df)
    signal = predict(df, model)
    price = get_price()

    if price is None:
        bot.send_message(CHAT_ID, "Ошибка получения цены.")
        return

    side = "Buy" if signal == 1 else "Sell"
    qty = round((CAPITAL * LEVERAGE) / price, 3)
    tp = round(price * (1 + TP_PCT / 100), 2) if side == "Buy" else round(price * (1 - TP_PCT / 100), 2)
    sl = round(price * (1 - SL_PCT / 100), 2) if side == "Buy" else round(price * (1 + SL_PCT / 100), 2)

    print("qty:", qty)
    print("tp:", tp)
    print("sl:", sl)

    result = place_order(side, qty, tp, sl)
    print("RAW Bybit result:", result)

    profit = abs(tp - price) * qty
    ret_msg = result.get("retMsg", "нет ответа") if isinstance(result, dict) else "ошибка при размещении ордера"

    message = (
        "Сигнал: {}\n"
        "Текущая цена входа: ${:.2f}\n"
        "Ожидаемая цена выхода: ${:.2f}\n"
        "Ожидаемая прибыль: ${:.2f}\n"
        "Точность прогноза: {:.2f}%\n"
        "Ответ от Bybit: {}".format(
            side, price, tp, profit, acc * 100, ret_msg
        )
    )
    bot.send_message(CHAT_ID, message)
@bot.message_handler(commands=["start"])
def handle_start(message):
    bot.send_message(message.chat.id, "Бот запущен. Прогноз + автотрейдинг каждые 5 мин.")
    def loop():
        while True:
            forecast_trade()
            time.sleep(300)
    threading.Thread(target=loop, daemon=True).start()

@bot.message_handler(commands=["once"])
def handle_once(message):
    forecast_trade()

@bot.message_handler(commands=["stop"])
def handle_stop(message):
    bot.send_message(message.chat.id, "Бот остановлен.")

bot.polling()
