"""
Ingestion & Defensive Parsing Layer.
Validates file types, handles encoding fallbacks, and safely parses CSV/Excel files into Pandas DataFrames.
"""

from __future__ import annotations
import io
from pathlib import Path
from typing import BinaryIO, Union, Tuple, Dict, Any
import pandas as pd


class IngestionError(Exception):
    """Custom exception raised when file validation or ingestion fails."""
    pass


def load_dataset(
    file_source: Union[str, Path, BinaryIO, bytes],
    filename: str | None = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Safely reads CSV or Excel files from disk, binary streams, or raw bytes.
    
    Args:
        file_source: Path to file, file-like binary stream (e.g. UploadFile.file), or raw bytes.
        filename: Optional filename hint (essential when passing binary streams).
        
    Returns:
        Tuple of (pd.DataFrame, metadata_dict).
        
    Raises:
        IngestionError: If file type is unsupported, corrupt, or empty.
    """
    # 1. Resolve file extension and name
    if isinstance(file_source, (str, Path)):
        path = Path(file_source)
        resolved_name = path.name
        resolved_ext = path.suffix.lower()
        file_bytes = path.read_bytes()
    else:
        resolved_name = filename or "uploaded_dataset.csv"
        resolved_ext = Path(resolved_name).suffix.lower()
        if isinstance(file_source, bytes):
            file_bytes = file_source
        elif hasattr(file_source, "read"):
            file_bytes = file_source.read()
            # If the stream was read, reset pointer if seekable
            if hasattr(file_source, "seek"):
                file_source.seek(0)
        else:
            raise IngestionError(f"Unsupported file source type: {type(file_source)}")

    if not file_bytes:
        raise IngestionError("The supplied file is empty (0 bytes).")

    size_kb = round(len(file_bytes) / 1024, 2)

    # 2. Parse based on file format
    buffer = io.BytesIO(file_bytes)
    df: pd.DataFrame | None = None

    if resolved_ext in (".csv", ".txt", ".tsv"):
        encodings_to_try = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
        delimiter = "\t" if resolved_ext == ".tsv" else None  # None triggers autodetection

        for encoding in encodings_to_try:
            try:
                buffer.seek(0)
                # If delimiter is None, pandas will auto-sniff with python engine
                if delimiter is None:
                    try:
                        df = pd.read_csv(buffer, encoding=encoding)
                    except (pd.errors.ParserError, UnicodeDecodeError):
                        buffer.seek(0)
                        df = pd.read_csv(buffer, sep=None, engine="python", encoding=encoding)
                else:
                    df = pd.read_csv(buffer, sep=delimiter, encoding=encoding)
                break
            except (UnicodeDecodeError, pd.errors.EmptyDataError):
                continue
            except Exception as e:
                # If it's a parse error under this encoding, try next
                continue

        if df is None:
            raise IngestionError(
                f"Failed to parse CSV file '{resolved_name}'. Ensure it has valid text encoding and delimiters."
            )

    elif resolved_ext in (".xlsx", ".xls"):
        try:
            buffer.seek(0)
            df = pd.read_excel(buffer, engine="openpyxl")
        except Exception as e:
            raise IngestionError(f"Failed to parse Excel file '{resolved_name}': {str(e)}")

    else:
        raise IngestionError(
            f"Unsupported file format '{resolved_ext}'. Please upload a CSV (.csv, .tsv) or Excel (.xlsx) file."
        )

    # 3. Structural validation
    if df.empty:
        raise IngestionError(f"Dataset '{resolved_name}' parsed successfully, but contains no rows.")
        
    if df.shape[1] == 0:
        raise IngestionError(f"Dataset '{resolved_name}' has 0 columns.")

    # 4. Clean column headers: strip whitespace
    df.columns = [str(col).strip() for col in df.columns]

    metadata: Dict[str, Any] = {
        "filename": resolved_name,
        "extension": resolved_ext,
        "file_size_kb": size_kb,
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
        "memory_usage_kb": round(float(df.memory_usage(deep=True).sum() / 1024), 2),
    }

    return df, metadata
