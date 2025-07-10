import pandas as pd
import numpy as np
import xgboost as xgb
import clickhouse_connect
from sklearn.metrics import mean_absolute_error, r2_score


class TSXMaxReturnPredictor:
    def __init__(self, clickhouse_params=None):
        self.model = None
        self.features = []
        self.client = None

        if clickhouse_params:
            self.client = clickhouse_connect.get_client(
                host=clickhouse_params['host'],
                port=clickhouse_params['port'],
                username=clickhouse_params['username'],
                password=clickhouse_params['password'],
                database=clickhouse_params.get('database', 'default')
            )

    # ✅ Fetch data from ClickHouse
    def fetch_data_from_clickhouse(self, table_name='tsx_eod'):
        if not self.client:
            raise Exception("ClickHouse client not initialized!")

        df = self.client.query_df(f"""
            SELECT *
            FROM {table_name}
        """)
        print(f"✅ Data fetched: {df.shape[0]} rows, {df['date'].min()} to {df['date'].max()}")
        return df

    # ✅ Data Cleaning
    def prepare_data(self, df):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['code', 'date'])
        df = df.drop(columns=['created_on'], errors='ignore')
        return df

    # ✅ Feature Engineering
    def create_features(self, df):
        df = df.copy()

        df['intraday_return'] = (df['close'] - df['open']) / df['open']
        df['volatility'] = (df['high'] - df['low']) / df['open']
        df['close_to_high'] = (df['high'] - df['close']) / df['open']
        df['close_to_low'] = (df['close'] - df['low']) / df['open']

        df['prev_close'] = df.groupby('code')['close'].shift(1)
        df['prev_open'] = df.groupby('code')['open'].shift(1)
        df['prev_return'] = (df['prev_close'] - df['prev_open']) / df['prev_open']

        window = 5
        df['rolling_volatility'] = df.groupby('code')['volatility'].transform(lambda x: x.rolling(window).std())
        df['rolling_return_mean'] = df.groupby('code')['intraday_return'].transform(lambda x: x.rolling(window).mean())
        df['rolling_return_std'] = df.groupby('code')['intraday_return'].transform(lambda x: x.rolling(window).std())

        df['log_volume'] = np.log1p(df['volume'])
        df['rolling_volume_mean'] = df.groupby('code')['log_volume'].transform(lambda x: x.rolling(window).mean())

        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month

        self.features = [
            'intraday_return',
            'volatility',
            'close_to_high',
            'close_to_low',
            'prev_return',
            'rolling_volatility',
            'rolling_return_mean',
            'rolling_return_std',
            'log_volume',
            'rolling_volume_mean',
            'day_of_week',
            'month'
        ]

        return df

    # ✅ Target Creation
    def create_target(self, df):
        df = df.copy()
        df['next_open'] = df.groupby('code')['open'].shift(-1)
        df['next_high'] = df.groupby('code')['high'].shift(-1)

        df['target'] = (df['next_high'] - df['next_open']) / df['next_open']
        df = df.dropna(subset=self.features + ['target'])
        return df

    # ✅ SMAPE Calculation
    def smape(self, y_true, y_pred):
        return 100 * np.mean(
            2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
        )

    # ✅ Rolling Training Prediction
    def predict_with_rolling_training(self, df, start_date, end_date):
        df = df.copy()
        results = []

        date_range = pd.date_range(start=start_date, end=end_date)

        for pred_date in date_range:
            pred_date = pd.to_datetime(pred_date)
            train_date = pred_date - pd.Timedelta(days=1)

            train_df = df[df['date'] <= train_date]
            eval_df = df[df['date'] == pred_date]

            if eval_df.empty or train_df.empty:
                continue

            X_train = train_df[self.features]
            y_train = train_df['target']

            X_eval = eval_df[self.features]
            y_true = eval_df['target']

            # Fresh model for each day
            model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.03,
                objective='reg:squarederror',
                random_state=42
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_eval)

            temp_df = eval_df[['code']].copy()
            temp_df['date'] = pred_date
            temp_df['actual_return (%)'] = y_true.values * 100
            temp_df['predicted_return (%)'] = y_pred * 100

            temp_df['SMAPE (%)'] = 100 * 2 * np.abs(temp_df['predicted_return (%)'] - temp_df['actual_return (%)']) / \
                (np.abs(temp_df['actual_return (%)']) + np.abs(temp_df['predicted_return (%)']) + 1e-8)

            temp_df['MAE (%)'] = np.abs(temp_df['predicted_return (%)'] - temp_df['actual_return (%)'])

            results.append(temp_df)

        final_df = pd.concat(results, ignore_index=True)

        return final_df
