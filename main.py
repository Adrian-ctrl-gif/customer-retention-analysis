from src.processing import DataProcessor


def main():
    processor = DataProcessor()

    # При первом запуске выгружаем заказы из базы.
    # При следующих используем сохранённые исходные данные.
    orders_path = processor.data_dir / "afisha_orders.csv"

    if orders_path.exists():
        print("Читаем сохранённые заказы...")
        processor.read_orders()
    else:
        print("Загружаем заказы из базы...")
        processor.load_orders()

    processor.load_rates()
    processor.preprocess()
    profiles = processor.create_profiles()
    retention = processor.calculate_retention()
    hypotheses = processor.check_hypotheses()
    correlations = processor.calculate_correlations()

    cleaning = processor.cleaning_report

    print("\nАНАЛИЗ ПОЛЬЗОВАТЕЛЕЙ ЯНДЕКС АФИШИ")
    print("\nПодготовка данных")
    print(f"Исходных заказов: {cleaning['initial_orders']}")
    print(f"Удалено дублей: {cleaning['removed_duplicates']}")
    print(f"Удалено заказов выше порога: {cleaning['removed_outliers']}")
    print(f"Порог выручки: {cleaning['revenue_p99']:.2f} ₽")
    print(f"Осталось заказов: {cleaning['remaining_orders']}")
    print(f"Пользователей: {len(profiles)}")

    avg_revenue = (
        (profiles["avg_revenue_rub"] * profiles["total_orders"]).sum()
        / profiles["total_orders"].sum()
    )
    return_rate = profiles["is_two"].mean()

    print("\nОсновные показатели")
    print(f"Средняя выручка с заказа: {avg_revenue:.2f} ₽")
    print(f"Доля повторных покупателей: {return_rate:.2%}")
    print(
        "Медианное число заказов:",
        profiles["total_orders"].median(),
    )

    titles = {
        "first_event": "Тип первого мероприятия",
        "first_device": "Устройство первого заказа",
        "first_region": "Регион первого мероприятия — топ-10 по аудитории",
        "first_weekday": "День первой покупки: 0 — Пн, 6 — Вс",
    }

    for column, table in retention.items():
        print(f"\n{titles[column]}")
        print(
            table[["users", "return_pct", "difference_pp"]]
            .head(10)
            .round(2)
            .to_string()
        )

    print("\nСравнение продуктовых гипотез")
    print(hypotheses.round(2).to_string(index=False))

    for _, row in hypotheses.iterrows():
        difference = row["difference_pp"]

        if row["observed_direction_matches"]:
            conclusion = "наблюдаемое различие соответствует гипотезе"
        else:
            conclusion = "наблюдаемое различие не поддерживает гипотезу"

        print(
            f"{row['hypothesis']}: {difference:+.2f} п.п.; "
            f"{conclusion}."
        )

    print("\nСвязь признаков с количеством заказов: φk")
    print(correlations.round(3).to_string())

    print("\nРекомендации")
    event_table = retention["first_event"]
    candidates = event_table.loc[
        (event_table["users"] >= 100)
        & (event_table["return_rate"] > return_rate)
    ]

    if not candidates.empty:
        print(
            "Проверить персональные предложения для сегментов "
            "мероприятий с возвратом выше среднего:"
        )
        print(
            candidates[["users", "return_pct"]]
            .round(2)
            .to_string()
        )

    print(
        "Проверять эффект предложений A/B-тестами, "
        "учитывая стоимость привлечения и дополнительную выручку."
    )
    print(
        "Отдельно изучить сверхактивные аккаунты "
        "и происхождение нулевой и отрицательной выручки."
    )

    print("\nОграничения")
    print(
        "Возврат рассчитан за весь период выборки. "
        "У поздних пользователей было меньше времени для повторной покупки."
    )
    print(
        "Сравнение топ-5 регионов относится к объединённой группе, "
        "а не к каждому региону отдельно."
    )
    print(
        "Связи не доказывают причинность. Средние за всю историю "
        "нельзя использовать для прогноза сразу после первого заказа."
    )


if __name__ == "__main__":
    main()