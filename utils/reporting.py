import quantstats as qs
import pandas as pd
import os

class ReportGenerator:
    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_quantstats_report(self, strategy_results, ticker="SPY"):
        """
        strategy_results: DataFrame with strategy returns, index should be datetime.
        """
        if strategy_results.empty:
            print("No results to generate report.")
            return

        # Ensure it's a Series for quantstats
        returns = strategy_results['returns'] if 'returns' in strategy_results else strategy_results.iloc[:, 0]

        output_file = os.path.join(self.output_dir, f"{ticker}_report.html")
        qs.reports.html(returns, output=output_file, title=f"Options Trading Bot Report - {ticker}")
        print(f"Report generated: {output_file}")
        return output_file

    def save_trade_log(self, trades_df, filename="trade_log.csv"):
        output_file = os.path.join(self.output_dir, filename)
        trades_df.to_csv(output_file, index=False)
        print(f"Trade log saved: {output_file}")
