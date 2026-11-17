"""Parses uploaded invoice PDFs into InvoiceEntity records for the billing pipeline."""

from dataclasses import dataclass

from dummy_project_module.clients.bq import run_query
from dummy_project_module.utils.gcs import download_file


@dataclass
class InvoiceEntity:
    invoice_id: str
    amount_jpy: int
    vendor_name: str


def parse_invoice(bucket_name: str, blob_name: str, local_path: str) -> InvoiceEntity:
    download_file(bucket_name, blob_name, local_path)
    # ... domain-specific PDF parsing omitted for the fixture ...
    return InvoiceEntity(invoice_id="INV-0001", amount_jpy=12000, vendor_name="Acme Corp")


def _unused_lookup() -> None:
    run_query("my-project", "billing", "vendors", "SELECT * FROM {table} LIMIT 1")
