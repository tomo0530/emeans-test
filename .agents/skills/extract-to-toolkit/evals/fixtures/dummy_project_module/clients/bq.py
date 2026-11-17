"""Thin wrapper around google-cloud-bigquery for running parameterized queries."""

from google.cloud import bigquery

from dummy_project_module.clients.bq_utils import format_table_ref


def run_query(project: str, dataset: str, table: str, sql: str) -> bigquery.table.RowIterator:
    client = bigquery.Client(project=project)
    table_ref = format_table_ref(project, dataset, table)
    return client.query(sql.format(table=table_ref)).result()
