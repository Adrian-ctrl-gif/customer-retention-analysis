import os
from pathlib import Path

import pandas as pd
import phik
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


class DataProcessor:

    def __init__(self):
        self.project_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.project_dir / "data"
        self.data_dir.mkdir(exist_ok=True)

        load_dotenv(self.project_dir / ".env")

        self.orders = None
        self.rates = None
        self.profiles = None

    def load_orders(self):
        required_variables = [
            "DB_USER",
            "DB_PWD",
            "DB_HOST",
            "DB_PORT",
            "DB_NAME",
        ]

        missing = [
            name for name in required_variables
            if not os.getenv(name) or os.getenv(name) == "FIXME"
        ]

        if missing:
            raise ValueError(
                "Заполните переменные в .env: " + ", ".join(missing)
            )

        connection_url = URL.create(
            drivername="postgresql+psycopg2",
            username=os.getenv("DB_USER"),
            password=os.getenv("DB_PWD"),
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT")),
            database=os.getenv("DB_NAME"),
        )

        query = """
        SELECT
            p.user_id,
            p.device_type_canonical,
            p.order_id,
            p.created_dt_msk AS order_dt,
            p.created_ts_msk AS order_ts,
            p.currency_code,
            p.revenue,
            p.tickets_count,
            (
                p.created_dt_msk::date
                - LAG(p.created_dt_msk::date) OVER (
                    PARTITION BY p.user_id
                    ORDER BY p.created_ts_msk, p.order_id
                )
            )::integer AS days_since_prev,
            p.event_id,
            e.event_name_code AS event_name,
            e.event_type_main,
            p.service_name,
            r.region_name,
            c.city_name
        FROM afisha.purchases AS p
        INNER JOIN afisha.events AS e
            ON p.event_id = e.event_id
        LEFT JOIN afisha.city AS c
            ON e.city_id = c.city_id
        LEFT JOIN afisha.regions AS r
            ON c.region_id = r.region_id
        WHERE p.device_type_canonical IN ('mobile', 'desktop')
            AND e.event_type_main != 'фильм'
        ORDER BY p.user_id, p.created_ts_msk, p.order_id
        """

        engine = create_engine(
            connection_url,
            connect_args={
                "sslmode": "require",
                "connect_timeout": 30,
            },
        )

        try:
            with engine.connect() as connection:
                orders = pd.read_sql_query(
                    text(query),
                    con=connection,
                )
        finally:
            engine.dispose()

        for column in ["user_id", "order_id", "event_id"]:
            orders[column] = orders[column].astype("string")

        orders.to_csv(
            self.data_dir / "afisha_orders.csv",
            index=False,
            encoding="utf-8",
        )

        self.orders = orders
        return self.orders

    def read_orders(self):
        """Прочитать сохранённую исходную выгрузку."""
        orders_path = self.data_dir / "afisha_orders.csv"

        self.orders = pd.read_csv(
            orders_path,
            dtype={
                "user_id": "string",
                "order_id": "string",
                "event_id": "string",
            },
        )

        return self.orders

    def load_rates(self):
        """Прочитать курс рубля за 100 тенге."""
        rates_path = self.data_dir / "final_tickets_tenge_df.csv"
        rates = pd.read_csv(rates_path)

        rates["data"] = pd.to_datetime(
            rates["data"],
            format="%Y-%m-%d",
        )

        if rates["data"].isna().any():
            raise ValueError("В файле курсов отсутствуют даты.")

        if rates["data"].duplicated().any():
            raise ValueError("В файле курсов повторяются даты.")

        if not rates["nominal"].eq(100).all():
            raise ValueError("Курс должен быть указан за 100 тенге.")

        if rates["curs"].isna().any() or not rates["curs"].gt(0).all():
            raise ValueError(
                "Обнаружен пропущенный или неположительный курс."
            )

        self.rates = rates
        return self.rates

    def preprocess(self):
        """Подготовить заказы для анализа."""
        if self.orders is None or self.rates is None:
            raise ValueError("Сначала загрузите заказы и курсы валют.")

        # Исходную выгрузку сохраняем без изменений.
        df = self.orders.copy()
        initial_count = len(df)

        df["order_dt"] = pd.to_datetime(df["order_dt"]).dt.normalize()
        df["order_ts"] = pd.to_datetime(df["order_ts"])

        category_columns = [
            "device_type_canonical",
            "currency_code",
            "service_name",
            "event_type_main",
            "region_name",
            "city_name",
        ]

        for column in category_columns:
            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )

        df["currency_code"] = df["currency_code"].str.lower()

        if not df["currency_code"].isin(["rub", "kzt"]).all():
            raise ValueError("Обнаружена неизвестная или пропущенная валюта.")

        for column in ["revenue", "tickets_count", "days_since_prev"]:
            df[column] = pd.to_numeric(df[column], errors="raise")

        # Курс указан в рублях за 100 тенге.
        rate_by_date = self.rates.set_index("data")["curs"]
        order_rates = df["order_dt"].map(rate_by_date)
        kzt_mask = df["currency_code"].eq("kzt")

        if order_rates.loc[kzt_mask].isna().any():
            raise ValueError("Не найден курс для части заказов в тенге.")

        df["revenue_rub"] = df["revenue"].astype(float)
        df.loc[kzt_mask, "revenue_rub"] = (
            df.loc[kzt_mask, "revenue"]
            * order_rates.loc[kzt_mask]
            / 100
        )

        # Ищем одинаковые по содержанию заказы без учёта order_id.
        duplicate_columns = [
            "user_id",
            "order_ts",
            "device_type_canonical",
            "service_name",
            "event_id",
            "currency_code",
            "revenue",
            "tickets_count",
        ]

        df = df.drop_duplicates(
            subset=duplicate_columns,
            keep="first",
        ).copy()

        removed_duplicates = initial_count - len(df)

        # Порог рассчитываем после удаления дублей.
        revenue_p99 = df["revenue_rub"].quantile(0.99)
        before_filter = len(df)

        df = df.loc[df["revenue_rub"] <= revenue_p99].copy()
        removed_outliers = before_filter - len(df)

        for column in category_columns:
            df[column] = df[column].astype("category")

        df["tickets_count"] = pd.to_numeric(
            df["tickets_count"],
            downcast="integer",
        )
        df["days_since_prev"] = df["days_since_prev"].astype("Int32")

        # После удаления заказов пересчитываем интервалы.
        df = df.sort_values(
            ["user_id", "order_ts", "order_id"]
        ).reset_index(drop=True)

        df["days_since_prev_clean"] = (
            df.groupby("user_id")["order_dt"].diff().dt.days
        )

        self.clean_orders = df
        self.cleaning_report = {
            "initial_orders": initial_count,
            "removed_duplicates": removed_duplicates,
            "removed_outliers": removed_outliers,
            "revenue_p99": revenue_p99,
            "remaining_orders": len(df),
            "remaining_users": df["user_id"].nunique(),
        }

        return self.clean_orders


    def create_profiles(self):
        """Создать одну строку с показателями на пользователя."""
        if getattr(self, "clean_orders", None) is None:
            raise ValueError("Сначала выполните preprocess().")

        df = self.clean_orders.sort_values(
            ["user_id", "order_ts", "order_id"]
        )

        profiles = df.groupby("user_id").agg(
            first_order_dt=("order_dt", "min"),
            last_order_dt=("order_dt", "max"),
            total_orders=("order_id", "nunique"),
            avg_revenue_rub=("revenue_rub", "mean"),
            avg_days_between=("days_since_prev_clean", "mean"),
            avg_tickets_count=("tickets_count", "mean"),
        )

        # Характеристики берём из одной строки первого заказа.
        first_orders = (
            df.drop_duplicates("user_id", keep="first")
            .set_index("user_id")
        )

        first_features = first_orders[
            [
                "device_type_canonical",
                "region_name",
                "event_type_main",
            ]
        ].rename(
            columns={
                "device_type_canonical": "first_device",
                "region_name": "first_region",
                "event_type_main": "first_event",
            }
        )

        profiles = profiles.join(first_features)

        profiles["is_two"] = (
            profiles["total_orders"] >= 2
        ).astype("int8")

        profiles["first_weekday"] = (
            profiles["first_order_dt"].dt.day_of_week
        )

        self.profiles = profiles
        return self.profiles

    def calculate_retention(self):
        """Рассчитать долю повторных покупателей по сегментам."""
        if self.profiles is None:
            raise ValueError("Сначала выполните create_profiles().")

        segment_columns = [
            "first_event",
            "first_device",
            "first_region",
            "first_weekday",
        ]

        baseline = self.profiles["is_two"].mean()
        results = {}

        for column in segment_columns:
            table = (
                self.profiles.groupby(column, observed=True)
                .agg(
                    users=("is_two", "size"),
                    returned_users=("is_two", "sum"),
                    return_rate=("is_two", "mean"),
                )
            )

            table["share_pct"] = (
                table["users"] / len(self.profiles) * 100
            )
            table["return_pct"] = table["return_rate"] * 100
            table["difference_pp"] = (
                table["return_rate"] - baseline
            ) * 100

            if column == "first_weekday":
                table = table.sort_index()
            else:
                table = table.sort_values("users", ascending=False)

            results[column] = table

        self.retention = results
        return self.retention

    def check_hypotheses(self):
        """Сравнить возврат для двух продуктовых гипотез."""
        if self.profiles is None:
            raise ValueError("Сначала выполните create_profiles().")

        profiles = self.profiles

        sport = profiles.loc[
            profiles["first_event"] == "спорт", "is_two"
        ]
        concerts = profiles.loc[
            profiles["first_event"] == "концерты", "is_two"
        ]

        # Активность региона определяем по числу пользователей.
        region_sizes = (
            profiles.groupby("first_region", observed=True)
            .size()
            .sort_values(ascending=False)
        )

        self.top_regions = region_sizes.head(5).index.tolist()
        top_mask = profiles["first_region"].isin(self.top_regions)

        top_users = profiles.loc[top_mask, "is_two"]
        other_users = profiles.loc[~top_mask, "is_two"]

        comparisons = [
            ("Спорт > концерты", sport, concerts),
            ("Топ-5 регионов > остальные", top_users, other_users),
        ]

        results = []

        for name, group_a, group_b in comparisons:
            if group_a.empty or group_b.empty:
                raise ValueError(
                    f"Для сравнения «{name}» одна из групп пуста."
                )

            rate_a = group_a.mean()
            rate_b = group_b.mean()

            results.append({
                "hypothesis": name,
                "users_a": len(group_a),
                "users_b": len(group_b),
                "return_a_pct": rate_a * 100,
                "return_b_pct": rate_b * 100,
                "difference_pp": (rate_a - rate_b) * 100,
                "observed_direction_matches": bool(rate_a > rate_b),
            })

        self.hypotheses = pd.DataFrame(results)
        return self.hypotheses

    def calculate_correlations(self):
        """Рассчитать связи признаков с количеством заказов."""
        if self.profiles is None:
            raise ValueError("Сначала выполните create_profiles().")

        data = self.profiles.copy()

        features = [
            "first_device",
            "first_region",
            "first_event",
            "first_weekday",
            "avg_revenue_rub",
            "avg_tickets_count",
        ]

        numeric_features = [
            "avg_revenue_rub",
            "avg_tickets_count",
        ]

        # День недели — категория, а не непрерывная величина.
        data["first_weekday"] = data["first_weekday"].astype("string")

        raw_matrix = data[
            ["total_orders"] + features
        ].phik_matrix(
            interval_cols=["total_orders"] + numeric_features,
            njobs=1,
        )

        data["order_segment"] = pd.cut(
            data["total_orders"],
            bins=[0, 1, 2, 4, float("inf")],
            labels=["1 заказ", "2 заказа", "3–4 заказа", "5 и более"],
        )

        grouped_matrix = data[
            ["order_segment"] + features
        ].phik_matrix(
            interval_cols=numeric_features,
            njobs=1,
        )

        comparison = pd.DataFrame({
            "total_orders": raw_matrix.loc[features, "total_orders"],
            "order_segment": grouped_matrix.loc[
                features, "order_segment"
            ],
        }).sort_values("order_segment", ascending=False)

        self.correlations = comparison
        return self.correlations