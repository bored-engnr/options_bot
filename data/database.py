import sqlite3
import pandas as pd
import logging

class OptionsDB:
    def __init__(self, db_path="options_data.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Table for historical stock prices
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS historical_prices (
                    symbol TEXT,
                    timestamp DATETIME,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (symbol, timestamp)
                )
            """)
            # Table for options chains
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS options_chains (
                    symbol TEXT,
                    expiration DATETIME,
                    strike REAL,
                    option_type TEXT,
                    last_price REAL,
                    bid REAL,
                    ask REAL,
                    volume INTEGER,
                    open_interest INTEGER,
                    implied_volatility REAL,
                    timestamp DATETIME,
                    PRIMARY KEY (symbol, expiration, strike, option_type, timestamp)
                )
            """)
            conn.commit()

    def save_historical_prices(self, symbol, df):
        if df.empty:
            return
        df = df.copy()
        df['symbol'] = symbol
        df = df.reset_index()
        df = df.rename(columns={'Date': 'timestamp', 'Datetime': 'timestamp'})
        
        allowed_cols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        df.columns = [c.lower() for c in df.columns]
        df = df[[c for c in allowed_cols if c in df.columns]]
        
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql('temp_historical', conn, if_exists='replace', index=False)
            conn.execute("""
                INSERT OR IGNORE INTO historical_prices (symbol, timestamp, open, high, low, close, volume)
                SELECT symbol, timestamp, open, high, low, close, volume FROM temp_historical
            """)
            conn.execute("DROP TABLE temp_historical")
            conn.commit()
            
    def save_options_chain(self, symbol, expiration, df):
        if df.empty:
            return
        df = df.copy()
        df['symbol'] = symbol
        df['expiration'] = expiration
        df['timestamp'] = pd.Timestamp.now()
        
        allowed_cols = ['symbol', 'expiration', 'strike', 'option_type', 'last_price', 'bid', 'ask', 'volume', 'open_interest', 'implied_volatility', 'timestamp']
        df = df[[c for c in allowed_cols if c in df.columns]]
        
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql('temp_options', conn, if_exists='replace', index=False)
            conn.execute(f"""
                INSERT OR IGNORE INTO options_chains ({", ".join(allowed_cols)})
                SELECT {", ".join(allowed_cols)} FROM temp_options
            """)
            conn.execute("DROP TABLE temp_options")
            conn.commit()

    def get_historical_prices(self, symbol, start_date=None, end_date=None):
        params = [symbol]
        query = "SELECT * FROM historical_prices WHERE symbol = ?"
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(query, conn, params=params, parse_dates=['timestamp'])

    def get_latest_options_chain(self, symbol, expiration=None):
        params = [symbol]
        query = "SELECT * FROM options_chains WHERE symbol = ?"
        if expiration:
            query += " AND expiration = ?"
            params.append(expiration)
        query += " ORDER BY timestamp DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(query, conn, params=params, parse_dates=['expiration', 'timestamp'])
            if not df.empty:
                latest_ts = df['timestamp'].max()
                return df[df['timestamp'] == latest_ts]
            return df
