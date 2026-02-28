import os
import re
import shutil
from pathlib import Path
from typing import Iterable, Any, Dict, List, Optional

def extract_year_from_filename(filename: str) -> Optional[str]:
    """
    Extract year from filename.
    Looks for 4-digit years (2015-2025) in filename.
    """
    match = re.search(r'(20[12][0-9])', filename)
    if match:
        return match.group(1)
    return None

def save_files(
    files: Iterable[Any],
    dest_dir: Path,
    year_mapping: Optional[Dict[str, int]] = None,
) -> List[Dict[str, object]]:
    """
    Save uploaded files to dest_dir, optionally appending a mapped year to the filename
    if the filename doesn't already contain a year.

    Expects each item in `files` to have:
      - .filename (str)
      - .file (binary file-like object)

    Returns a list of dicts: {filename, path, year}
    """
    if extract_year_from_filename is None:
        raise ValueError("extract_year_from_filename must be provided")

    year_mapping = year_mapping or {}
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Dict[str, object]] = []
    used_names: Dict[str, int] = {}

    def _unique_name(name: str) -> str:
        """Avoid overwriting if duplicates appear in the upload set."""
        base, ext = os.path.splitext(name)
        if name not in used_names:
            used_names[name] = 0
            return name
        used_names[name] += 1
        return f"{base}__{used_names[name]}{ext}"

    for fobj in files:
        original = fobj.filename
        filename = original

        # If year mapping provided, rename file to include year (only if missing)
        if original in year_mapping and not extract_year_from_filename(filename):
            year = year_mapping[original]
            base, ext = os.path.splitext(filename)
            filename = f"{base}_{year}{ext}"

        filename = _unique_name(filename)
        file_path = dest_dir / filename

        with open(file_path, "wb") as out:
            shutil.copyfileobj(fobj.file, out)

        saved.append(
            {
                "filename": filename,
                "path": str(file_path),
                "year": extract_year_from_filename(filename),
            }
        )

    return saved