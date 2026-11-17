"""Nightly batch entrypoint: ingest invoices, upload artifacts, and load results into BigQuery."""

from dummy_project_module.clients.bq import run_query
from dummy_project_module.domain.invoice_parser import parse_invoice
from dummy_project_module.utils.gcs import upload_file


def run_nightly_batch() -> None:
    invoice = parse_invoice("raw-bucket", "invoices/latest.pdf", "/tmp/latest.pdf")
    upload_file("processed-bucket", "/tmp/latest.pdf", f"invoices/{invoice.invoice_id}.pdf")
    run_query("my-project", "billing", "invoices", "SELECT * FROM {table} LIMIT 10")
