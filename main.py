from sklearn.preprocessing import MinMaxScaler
from plyer import notification
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import yfinance as yf
from pandas_datareader import data as pdr
import time
import sqlite3
from flask import Flask, render_template





#flask app
app = Flask(__name__, static_folder='statics')

# loading the model
model = joblib.load('model/model.pkl')

# fetching data from yahoo finance
end = datetime.now()
end = end - timedelta(days=3)  # for testing purposes
start = end - timedelta(days=160)
yf.pdr_override()


def get_data(ticker, start, end):
    df = pdr.get_data_yahoo('AAPL', start, end)
    return df


def get_data(ticker, start, end):
    df = pdr.get_data_yahoo('AAPL', start, end)
    return df




@app.route('/')
def home():
    today = datetime.today()
    prev_day = today - timedelta(days=1)
    is_pred = False
    while True:
        today = datetime.today()
        if not is_pred:
            df = get_data('AAPL', start, end)
            data = df.filter(['Close'])
            data = data.iloc[len(data)-60:]
            dataset = data.values
            prev = data.iloc[-1].values[0]
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(dataset)
            x_pred = np.reshape(scaled, (1, scaled.shape[0], 1))
            pred = model.predict(x_pred)
            pred_price = scaler.inverse_transform(pred)
            pred_price = np.reshape(pred_price, 1)[0]
            print("predicted price",pred_price)
            # store data in sqlite3 database
            conn = sqlite3.connect('stocks.db')
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS stocks (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT,previous REAL ,predicted REAL)")
            
            cursor.execute("INSERT INTO stocks (date, previous, predicted) VALUES (?, ?, ?)", (today, prev, pred_price))
            conn.commit()
            cursor.execute("SELECT * FROM stocks ORDER BY id DESC")
            data = cursor.fetchall()
            conn.close()


            if pred_price > prev:
                signal = "BUY"
            else:
                signal = "SELL"

            message = f"SIGNAL {signal}\n prev  {prev} \n pred {pred_price}"
            print(message)
            notification.notify(
                title="Apple stocks",
                message=message,
                timeout=20
            )
            is_pred = True #toggle the variable
            print(data)
            return render_template('index.html', data=data[:10])
        if today.day != prev_day.day:
            is_pred = False
            prev_day = today
        time.sleep(60)
        return render_template('index.html',data=data[:10])    





if __name__ == '__main__':
    app.run(debug=True)





