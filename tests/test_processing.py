"""Small, synthetic-data regression tests. No database or .env needed."""
import unittest

import pandas as pd

from src.processing import DataProcessor


class ProcessingTests(unittest.TestCase):
    def make_processor(self):
        # Bypass __init__ because it prepares local project directories and .env.
        processor = DataProcessor.__new__(DataProcessor)
        processor.orders = pd.DataFrame([
            dict(user_id="A", order_id="1", event_id="e1", order_dt="2024-06-01",
                 order_ts="2024-06-01 12:00:00", device_type_canonical="mobile",
                 currency_code="rub", revenue=100, tickets_count=1,
                 days_since_prev=None, service_name="afisha", event_type_main="спорт",
                 region_name="Москва", city_name="Москва"),
            dict(user_id="A", order_id="2", event_id="e2", order_dt="2024-06-03",
                 order_ts="2024-06-03 12:00:00", device_type_canonical="mobile",
                 currency_code="kzt", revenue=1000, tickets_count=2,
                 days_since_prev=2, service_name="afisha", event_type_main="концерты",
                 region_name="Москва", city_name="Москва"),
            dict(user_id="B", order_id="3", event_id="e3", order_dt="2024-06-02",
                 order_ts="2024-06-02 12:00:00", device_type_canonical="desktop",
                 currency_code="rub", revenue=80, tickets_count=1,
                 days_since_prev=None, service_name="afisha", event_type_main="концерты",
                 region_name="СПб", city_name="СПб"),
            dict(user_id="C", order_id="4", event_id="e4", order_dt="2024-06-04",
                 order_ts="2024-06-04 12:00:00", device_type_canonical="mobile",
                 currency_code="rub", revenue=100000, tickets_count=1,
                 days_since_prev=None, service_name="afisha", event_type_main="спорт",
                 region_name="Москва", city_name="Москва"),
        ])
        processor.rates = pd.DataFrame({
            "data": pd.to_datetime(["2024-06-01", "2024-06-02",
                                    "2024-06-03", "2024-06-04"]),
            "curs": [7.5, 7.5, 7.5, 7.5],
        })
        return processor

    def test_cleanup_and_conversion(self):
        processor = self.make_processor()
        cleaned = processor.preprocess()
        # The unusually large order is excluded by the 99th-percentile rule.
        self.assertEqual(len(cleaned), 3)
        self.assertEqual(processor.cleaning_report["removed_outliers"], 1)
        converted = cleaned.loc[cleaned["order_id"] == "2", "revenue_rub"].iloc[0]
        self.assertAlmostEqual(converted, 75.0)

    def test_repeat_buyers_and_intervals(self):
        processor = self.make_processor()
        processor.preprocess()
        profiles = processor.create_profiles()
        self.assertEqual(len(profiles), 2)
        self.assertEqual(int(profiles.loc["A", "total_orders"]), 2)
        self.assertEqual(int(profiles.loc["A", "is_two"]), 1)
        self.assertEqual(int(profiles.loc["B", "is_two"]), 0)
        interval = processor.clean_orders.loc[
            processor.clean_orders["order_id"] == "2", "days_since_prev_clean"
        ].iloc[0]
        self.assertEqual(int(interval), 2)
        retention = processor.calculate_retention()
        self.assertIn("first_event", retention)
        self.assertAlmostEqual(profiles["is_two"].mean(), 0.5)


if __name__ == "__main__":
    unittest.main()
