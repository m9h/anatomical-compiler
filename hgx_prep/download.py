"""Data acquisition from GEO and Zenodo."""
from __future__ import annotations
import gzip
import json
import shutil
from pathlib import Path
from urllib.request import urlopen, urlretrieve, Request


def _progress_hook(count, block_size, total_size):
    pct = count * block_size * 100 / total_size if total_size > 0 else 0
    mb = count * block_size / 1e6
    total_mb = total_size / 1e6
    print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct:.0f}%)", end="", flush=True)


def download_file(url: str, dest: Path, *, desc: str = "") -> bool:
    """Download a file with progress. Try HTTPS, fall back to FTP if URL has ftp://."""
    label = desc or dest.name
    print(f"  Downloading {label} ...")
    print(f"    {url}")
    try:
        urlretrieve(url, str(dest), reporthook=_progress_hook)
        print()
        return True
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return False


def download_geo(
    geo_id: str,
    dest_dir: Path,
    files: list[str] | None = None,
) -> list[Path]:
    """Download supplementary files from a GEO accession.

    If files is None, downloads the series matrix (not supplementary).
    Otherwise, downloads each named file from the supplementary URL.

    Returns list of successfully downloaded file paths.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    base_https = f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={geo_id}&format=file&file="
    base_ftp = f"ftp://ftp.ncbi.nlm.nih.gov/geo/series/{geo_id[:7]}nnn/{geo_id}/suppl"

    downloaded = []
    for filename in (files or []):
        dest = dest_dir / filename
        if dest.exists():
            print(f"  {filename} already exists, skipping")
            downloaded.append(dest)
            continue

        ok = download_file(f"{base_https}{filename}", dest, desc=filename)
        if not ok:
            print("  Retrying via FTP ...")
            ok = download_file(f"{base_ftp}/{filename}", dest, desc=filename)

        if ok:
            # Decompress .gz if needed
            if dest.suffix == ".gz" and dest.exists():
                out = dest.with_suffix("")
                decompress_gz(dest, out)
                downloaded.append(out)
            else:
                downloaded.append(dest)
        else:
            print(f"  FAILED: {filename}")

    return downloaded


def resolve_zenodo_doi(doi: str) -> dict | None:
    """Resolve a Zenodo DOI to a record with file URLs.

    Returns dict with 'record_id' and 'files' list, or None.
    Works for DOIs like '10.5281/zenodo.5242913'.
    """
    if "zenodo" not in doi:
        return None

    # Extract record ID from DOI
    parts = doi.split("zenodo.")
    if len(parts) < 2:
        return None
    record_id = parts[-1].strip("/")

    url = f"https://zenodo.org/api/records/{record_id}"
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        files = []
        for f in data.get("files", []):
            files.append({
                "filename": f["key"],
                "size": f["size"],
                "url": f["links"]["self"],
                "checksum": f.get("checksum", ""),
            })

        return {
            "record_id": record_id,
            "title": data.get("metadata", {}).get("title", ""),
            "files": files,
        }
    except Exception as e:
        print(f"  Could not resolve Zenodo DOI: {e}")
        return None


def decompress_gz(gz_path: Path, out_path: Path) -> None:
    """Decompress a .gz file and remove the original."""
    print(f"  Decompressing -> {out_path.name}")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()
