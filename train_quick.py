# train_quick.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("DailyDelhiClimateTrain.csv")

# rename to standard columns used by the app; adjust if your dataset differs
df = df.rename(columns={
    'meantemp': 'Temp',
    'humidity': 'Humidity',
    'wind_speed': 'WindSpeed',
    'meanpressure': 'Pressure'
})

df['temp_prev'] = df['Temp'].shift(1)
df = df.dropna()

X = df[['Humidity','WindSpeed','Pressure','temp_prev']]
y = df['Temp']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
model.fit(x_train, y_train)

joblib.dump(model, "weather_model.pkl")
print("Saved weather_model.pkl")


