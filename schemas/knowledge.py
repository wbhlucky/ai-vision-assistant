from pydantic import BaseModel


class KnowledgeIngestResponse(BaseModel):
    scan_path: str
    scanned_files: int
    skipped_existing: int
    ingested_files: int
    errors: int
    bm25_rebuilt: bool
    has_new_data: bool

