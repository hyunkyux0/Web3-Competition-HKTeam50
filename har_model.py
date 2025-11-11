# har_model.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class HARModel:
    """
    Heterogeneous Autoregressive (HAR) Model for Realized Variance forecasting.
    
    Implements both regular HAR and log-HAR models as described in:
    Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized Volatility"
    """
    
    def __init__(self, use_log=False):
        """
        Initialize HAR model.
        
        Parameters:
        use_log (bool): If True, uses log-HAR model, otherwise uses regular HAR
        """
        self.use_log = use_log
        self.model = LinearRegression()
        self.is_fitted = False
        self.phi_0 = None  # Intercept
        self.phi_D = None  # Daily coefficient
        self.phi_W = None  # Weekly coefficient
        self.phi_M = None  # Monthly coefficient
        
    def calculate_realized_variance(self, prices, frequency='15min'):
        """
        Calculate realized variance from high-frequency price data.
        
        RV_t = sum(R_{t+j/m}^2) for j=1 to m
        where R_{t+j/m} is the intraday return
        
        Parameters:
        prices (pd.Series): Price series with datetime index
        frequency (str): Data frequency ('15min', '10min', '30min', etc.)
        
        Returns:
        pd.Series: Daily realized variance
        """
        # Calculate intraday returns
        returns = np.log(prices / prices.shift(1))
        returns = returns.dropna()
        
        # Square the returns
        squared_returns = returns ** 2
        
        # Group by date and sum to get daily realized variance
        if isinstance(squared_returns.index, pd.DatetimeIndex):
            daily_rv = squared_returns.groupby(squared_returns.index.date).sum()
            daily_rv.index = pd.to_datetime(daily_rv.index)
        else:
            # Fallback if index is not datetime
            daily_rv = squared_returns.resample('D').sum()
        
        return daily_rv
    
    def calculate_rv_components(self, daily_rv):
        """
        Calculate HAR model components: RV_D, RV_W, RV_M
        
        Parameters:
        daily_rv (pd.Series): Daily realized variance series
        
        Returns:
        pd.DataFrame: DataFrame with RV_D, RV_W, RV_M columns
        """
        df = pd.DataFrame(index=daily_rv.index)
        
        # RV_D: Daily realized variance (lagged by 1 day)
        df['RV_D'] = daily_rv.shift(1)
        
        # RV_W: Average realized variance over previous 7 days
        df['RV_W'] = daily_rv.rolling(window=3).mean().shift(1)
        
        # RV_M: Average realized variance over previous 30 days
        df['RV_M'] = daily_rv.rolling(window=5).mean().shift(1)
        
        # Target: Next day's realized variance
        df['RV_target'] = daily_rv
        
        # Remove NaN values
        df = df.dropna()
        
        return df
    
    def fit(self, prices, min_training_days=3):
        """
        Fit the HAR model to historical price data.
        
        Parameters:
        prices (pd.Series): Price series with datetime index
        min_training_days (int): Minimum number of days needed for training
        
        Returns:
        dict: Model coefficients and statistics
        """
        # Step 1: Calculate daily realized variance
        print("Calculating realized variance from 15-minute returns...")
        daily_rv = self.calculate_realized_variance(prices, frequency='15min')
        
        if len(daily_rv) < min_training_days:
            raise ValueError(f"Insufficient data: need at least {min_training_days} days, got {len(daily_rv)}")
        
        # Step 2: Calculate HAR components
        print("Calculating HAR components (daily, weekly, monthly)...")
        har_data = self.calculate_rv_components(daily_rv)
        
        if len(har_data) < min_training_days:
            raise ValueError(f"Insufficient data after calculating components: need {min_training_days} days")
        
        # Step 3: Prepare features and target
        X = har_data[['RV_D', 'RV_W', 'RV_M']].values
        y = har_data['RV_target'].values
        
        # Apply log transformation if using log-HAR model
        if self.use_log:
            print("Fitting log-HAR model...")
            # Add small constant to avoid log(0)
            X = np.log(X + 1e-10)
            y = np.log(y + 1e-10)
        else:
            print("Fitting regular HAR model...")
        
        # Step 4: Fit the model
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Store coefficients
        self.phi_0 = self.model.intercept_
        self.phi_D = self.model.coef_[0]
        self.phi_W = self.model.coef_[1]
        self.phi_M = self.model.coef_[2]
        
        # Calculate R-squared
        y_pred = self.model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Store the daily RV for forecasting
        self.daily_rv = daily_rv
        self.har_data = har_data
        
        print(f"✓ HAR model fitted successfully!")
        print(f"  Training samples: {len(X)}")
        print(f"  R-squared: {r_squared:.4f}")
        
        return {
            'phi_0': self.phi_0,
            'phi_D': self.phi_D,
            'phi_W': self.phi_W,
            'phi_M': self.phi_M,
            'r_squared': r_squared,
            'n_samples': len(X),
            'use_log': self.use_log
        }
    
    def forecast(self, rv_d, rv_w, rv_m, horizon=1):
        """
        Forecast future realized variance.
        
        Parameters:
        rv_d (float): Daily realized variance (previous day)
        rv_w (float): Weekly average realized variance (previous 7 days)
        rv_m (float): Monthly average realized variance (previous 30 days)
        horizon (int): Forecast horizon in days (default=1)
        
        Returns:
        float: Forecasted realized variance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        # Prepare features
        X = np.array([[rv_d, rv_w, rv_m]])
        
        if self.use_log:
            X = np.log(X + 1e-10)
        
        # Make prediction
        if self.use_log:
            log_forecast = self.model.predict(X)[0]
            forecast = np.exp(log_forecast)
        else:
            forecast = self.model.predict(X)[0]
        
        # For multi-step forecasts (simple iterative approach)
        if horizon > 1:
            forecasts = [forecast]
            for _ in range(horizon - 1):
                # Use previous forecast as new RV_D
                new_rv_w = (rv_d + sum(forecasts[-7:])) / min(8, len(forecasts) + 1)
                new_rv_m = (rv_d + sum(forecasts[-30:])) / min(31, len(forecasts) + 1)
                
                X_new = np.array([[forecasts[-1], new_rv_w, new_rv_m]])
                if self.use_log:
                    X_new = np.log(X_new + 1e-10)
                    log_forecast = self.model.predict(X_new)[0]
                    forecast = np.exp(log_forecast)
                else:
                    forecast = self.model.predict(X_new)[0]
                
                forecasts.append(forecast)
            
            return forecasts[-1]
        
        return forecast
    
    def forecast_volatility(self, rv_d, rv_w, rv_m, horizon=1, annualization_factor=None):
        """
        Forecast volatility (standard deviation) instead of variance.
        
        Parameters:
        rv_d, rv_w, rv_m: Realized variance components
        horizon (int): Forecast horizon
        annualization_factor (float): Factor to annualize volatility (e.g., sqrt(252) for daily)
        
        Returns:
        float: Forecasted volatility (standard deviation)
        """
        rv_forecast = self.forecast(rv_d, rv_w, rv_m, horizon)
        vol_forecast = np.sqrt(rv_forecast)
        
        if annualization_factor is not None:
            vol_forecast *= np.sqrt(annualization_factor)
        
        return vol_forecast
    
    def rolling_forecast(self, prices, train_window=90, forecast_horizon=1):
        """
        Generate rolling out-of-sample forecasts.
        
        Parameters:
        prices (pd.Series): Price series
        train_window (int): Number of days to use for training
        forecast_horizon (int): Days ahead to forecast
        
        Returns:
        pd.DataFrame: DataFrame with actual RV and forecasted RV
        """
        daily_rv = self.calculate_realized_variance(prices)
        har_data = self.calculate_rv_components(daily_rv)
        
        forecasts = []
        actuals = []
        dates = []
        
        min_start = 30 + train_window  # Need 30 days for RV_M calculation
        
        for i in range(min_start, len(har_data) - forecast_horizon):
            # Training data
            train_data = har_data.iloc[i-train_window:i]
            
            X_train = train_data[['RV_D', 'RV_W', 'RV_M']].values
            y_train = train_data['RV_target'].values
            
            if self.use_log:
                X_train = np.log(X_train + 1e-10)
                y_train = np.log(y_train + 1e-10)
            
            # Fit model
            temp_model = LinearRegression()
            temp_model.fit(X_train, y_train)
            
            # Forecast
            X_test = har_data.iloc[i][['RV_D', 'RV_W', 'RV_M']].values.reshape(1, -1)
            
            if self.use_log:
                X_test = np.log(X_test + 1e-10)
                forecast = np.exp(temp_model.predict(X_test)[0])
            else:
                forecast = temp_model.predict(X_test)[0]
            
            actual = har_data.iloc[i + forecast_horizon]['RV_target']
            
            forecasts.append(forecast)
            actuals.append(actual)
            dates.append(har_data.index[i + forecast_horizon])
        
        results = pd.DataFrame({
            'date': dates,
            'actual_rv': actuals,
            'forecast_rv': forecasts
        })
        results.set_index('date', inplace=True)
        
        # Calculate forecast accuracy metrics
        mse = np.mean((results['actual_rv'] - results['forecast_rv']) ** 2)
        mae = np.mean(np.abs(results['actual_rv'] - results['forecast_rv']))
        
        print(f"\nOut-of-sample forecast performance:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        
        return results
    
    def get_current_rv_components(self, prices):
        """
        Get current RV components for forecasting.
        
        Parameters:
        prices (pd.Series): Recent price data
        
        Returns:
        tuple: (rv_d, rv_w, rv_m)
        """
        daily_rv = self.calculate_realized_variance(prices)
        
        if len(daily_rv) < 30:
            raise ValueError("Need at least 30 days of data")
        
        rv_d = daily_rv.iloc[-1]
        rv_w = daily_rv.iloc[-7:].mean()
        rv_m = daily_rv.iloc[-30:].mean()
        
        return rv_d, rv_w, rv_m
    
    def print_model_summary(self):
        """Print summary of fitted model."""
        if not self.is_fitted:
            print("Model has not been fitted yet.")
            return
        
        model_type = "Log-HAR" if self.use_log else "HAR"
        
        print(f"\n{'='*60}")
        print(f"{model_type} Model Summary")
        print("="*60)
        print(f"Model Equation:")
        if self.use_log:
            print(f"  log(RV_{{t+1}}) = φ_0 + φ_D·log(RV_D) + φ_W·log(RV_W) + φ_M·log(RV_M)")
        else:
            print(f"  RV_{{t+1}} = φ_0 + φ_D·RV_D + φ_W·RV_W + φ_M·RV_M")
        
        print(f"\nEstimated Coefficients:")
        print(f"  φ_0 (Intercept)  : {self.phi_0:>10.6f}")
        print(f"  φ_D (Daily)      : {self.phi_D:>10.6f}")
        print(f"  φ_W (Weekly)     : {self.phi_W:>10.6f}")
        print(f"  φ_M (Monthly)    : {self.phi_M:>10.6f}")
        print("="*60)