def future_forecast(historical_data, future_years):
    historical_data = historical_data.dropna()
    #were dealing with stock prices float rep is the best option to maintain optimal data storage
    historical_data = historical_data.astype(float)
    #convert panda series into a numpy array creates compatiability with ARIMA model
    endog = np.asarray(historical_data)
    model = ARIMA(endog, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(future_years))
    return forecast

# Load and clean the data
file_path = 'Starbucks_StockPrice.csv'
stock_data = pd.read_csv(file_path)
stock_data = stock_data.dropna(axis=0)
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data['year'] = stock_data['Date'].dt.year

# the issue here was the focus on using the group.by() function
historical_open = stock_data.groupby('year')['Open'].mean().reset_index()
historical_adj_close = stock_data.groupby('year')['Adj Close'].mean().reset_index()
historical_volume = stock_data.groupby('year')['Volume'].sum().reset_index()
historical_high = stock_data.groupby('year')['High'].mean().reset_index()
historical_low = stock_data.groupby('year')['Low'].mean().reset_index()

# Forecast future values for each feature
future_years = [2025, 2026, 2027, 2028, 2029, 2030]
forecast_open = future_forecast(historical_open['Open'], future_years)
forecast_high = future_forecast(historical_high['High'], future_years)
forecast_low = future_forecast(historical_low['Low'], future_years)
