"""Time-series forecasting for independent variables.

Provides multiple forecaster backends:
  - ProphetForecaster: Facebook Prophet (additive trend + yearly seasonality)
  - LSTMForecaster: PyTorch LSTM (learns nonlinear temporal patterns)
  - RidgeForecaster: Polynomial + lag features (fast, stable baseline)
  - EnsembleForecaster: Weighted average of all three
"""

import warnings
import numpy as np
import pandas as pd
from config import config

warnings.filterwarnings("ignore")


class RidgeForecaster:
    """Fast polynomial trend + lag regression baseline."""

    def __init__(self):
        self.model = None
        self._history = None

    def _features(self, vals: np.ndarray) -> np.ndarray:
        n = len(vals)
        t = np.arange(n).reshape(-1, 1).astype(float)
        cols = [np.ones((n, 1)), t, t ** 2, np.sin(2 * np.pi * t / 10)]
        for lag in [1, 2, 3, 5]:
            lagged = np.full(n, np.nan)
            if n > lag:
                lagged[lag:] = vals[:-lag]
            cols.append(lagged.reshape(-1, 1))
        return np.hstack(cols)

    def fit(self, series: pd.Series) -> "RidgeForecaster":
        from sklearn.linear_model import Ridge
        vals = series.values.astype(float)
        X = self._features(vals)
        valid = ~np.isnan(X).any(axis=1)
        self.model = Ridge(alpha=0.5)
        self.model.fit(X[valid], vals[valid])
        self._history = vals.copy()
        return self

    def predict(self, steps: int) -> np.ndarray:
        hist = list(self._history)
        preds = []
        for _ in range(steps):
            arr = np.array(hist)
            X = self._features(arr)
            p = self.model.predict(X[-1:])[0]
            preds.append(p)
            hist.append(p)
        return np.array(preds)


class ProphetForecaster:
    """Facebook Prophet forecaster with trend + seasonality."""

    def __init__(self, yearly: bool = False):
        self.yearly = yearly
        self.model = None

    def fit(self, series: pd.Series) -> "ProphetForecaster":
        try:
            from prophet import Prophet
        except ImportError:
            self._fallback = RidgeForecaster()
            self._fallback.fit(series)
            return self

        df = pd.DataFrame({
            "ds": pd.date_range(
                start=f"{config.HIST_START}-01-01",
                periods=len(series), freq="YS"
            ),
            "y": series.values,
        })
        self.model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_mode="additive",
            yearly_seasonality=self.yearly,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        self.model.fit(df)
        return self

    def predict(self, steps: int) -> np.ndarray:
        if not hasattr(self, "model") or self.model is None:
            return self._fallback.predict(steps) if hasattr(self, "_fallback") else np.zeros(steps)
        try:
            future = self.model.make_future_dataframe(periods=steps, freq="YS")
            forecast = self.model.predict(future)
            return forecast["yhat"].values[-steps:]
        except Exception:
            return self._fallback.predict(steps) if hasattr(self, "_fallback") else np.zeros(steps)


class LSTMForecaster:
    """PyTorch LSTM forecaster for nonlinear temporal patterns."""

    def __init__(self, hidden_size: int = 32, seq_len: int = 5, epochs: int = 50):
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.epochs = epochs
        self.model = None
        self._scaler_mean = 0.0
        self._scaler_std = 1.0

    def _build_sequences(self, vals: np.ndarray) -> tuple:
        X, y = [], []
        for i in range(self.seq_len, len(vals)):
            X.append(vals[i - self.seq_len:i])
            y.append(vals[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def fit(self, series: pd.Series) -> "LSTMForecaster":
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            self._fallback = RidgeForecaster()
            self._fallback.fit(series)
            return self

        vals = series.values.astype(float)
        self._scaler_mean = np.mean(vals)
        self._scaler_std = max(np.std(vals), 1e-8)
        vals_norm = (vals - self._scaler_mean) / self._scaler_std

        if len(vals_norm) < self.seq_len + 5:
            self._fallback = RidgeForecaster()
            self._fallback.fit(series)
            return self

        X, y = self._build_sequences(vals_norm)
        X_tensor = torch.tensor(X).unsqueeze(-1)
        y_tensor = torch.tensor(y)

        class LSTMModel(nn.Module):
            def __init__(self, input_size=1, hidden_size=32):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=1)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                _, (h, _) = self.lstm(x)
                return self.fc(h.squeeze(0)).squeeze(-1)

        self.model = LSTMModel(hidden_size=self.hidden_size)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = self.model(X_tensor)
            loss = loss_fn(pred, y_tensor)
            loss.backward()
            optimizer.step()

        self.model.eval()
        self._history_norm = vals_norm.tolist()
        return self

    def predict(self, steps: int) -> np.ndarray:
        if not hasattr(self, "model") or self.model is None:
            return self._fallback.predict(steps) if hasattr(self, "_fallback") else np.zeros(steps)
        try:
            import torch
        except ImportError:
            return self._fallback.predict(steps) if hasattr(self, "_fallback") else np.zeros(steps)

        hist = list(self._history_norm)
        preds = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(steps):
                seq = torch.tensor(hist[-self.seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                p = self.model(seq).item()
                preds.append(p)
                hist.append(p)

        preds = np.array(preds) * self._scaler_std + self._scaler_mean
        return preds


class EnsembleForecaster:
    """Weighted ensemble of Prophet, LSTM, and Ridge forecasters."""

    def __init__(self, use_prophet: bool = True, use_lstm: bool = True):
        self.forecasters = {}
        self.weights = {}

        self.forecasters["ridge"] = RidgeForecaster()
        self.weights["ridge"] = 0.30

        if use_prophet:
            self.forecasters["prophet"] = ProphetForecaster()
            self.weights["prophet"] = 0.35

        if use_lstm:
            self.forecasters["lstm"] = LSTMForecaster(
                hidden_size=32, seq_len=5, epochs=40
            )
            self.weights["lstm"] = 0.35

        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def fit_predict(self, series: pd.Series, steps: int) -> np.ndarray:
        preds = {}
        for name, fc in self.forecasters.items():
            try:
                fc.fit(series)
                p = fc.predict(steps)
                preds[name] = p
            except Exception:
                continue

        if not preds:
            return np.zeros(steps)

        result = np.zeros(steps)
        total_w = 0.0
        for name, p in preds.items():
            w = self.weights.get(name, 1.0 / len(preds))
            result += w * p
            total_w += w
        return result / total_w if total_w > 0 else result


def forecast_variable(
    series: pd.Series,
    steps: int = None,
    use_prophet: bool = True,
    use_lstm: bool = True,
) -> np.ndarray:
    if steps is None:
        steps = config.FORECAST_END - config.HIST_END
    fc = EnsembleForecaster(use_prophet=use_prophet, use_lstm=use_lstm)
    return fc.fit_predict(series, steps)
