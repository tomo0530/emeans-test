"""Query-building helpers shared by the BigQuery client."""

from datetime import date


def format_table_ref(project: str, dataset: str, table: str) -> str:
    return f"`{project}.{dataset}.{table}`"


def date_partition_suffix(day: date) -> str:
    return day.strftime("%Y%m%d")
