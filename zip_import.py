from __future__ import annotations
import io
import zipfile
from pathlib import PurePosixPath
from typing import Dict, List, Tuple
import mimetypes

ALLOWED_EXTS = {".csv", ".xml", ".mvd2", ".xlsx"}

def _sanitize_name(name: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in (" ", "_", "-") else "_" for ch in name.strip())
    return out[:80] if out else "Group"

def _infer_mime(name: str) -> str:
    mt, _ = mimetypes.guess_type(name)
    return mt or "application/octet-stream"

def parse_zip_group_map(zip_bytes: bytes) -> Dict[str, List[Tuple[str, bytes]]]:
    groups: Dict[str, List[Tuple[str, bytes]]] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            path = PurePosixPath(info.filename)
            if any(part.startswith("__MACOSX") or part.startswith(".") for part in path.parts):
                continue
            ext = path.suffix.lower()
            if ext not in ALLOWED_EXTS:
                continue
            top = path.parts[0] if len(path.parts) > 1 else "root"
            group = _sanitize_name(top)
            with zf.open(info, "r") as fh:
                data = fh.read()
            groups.setdefault(group, []).append((path.name, data))
    return groups

def conditions_from_zip_bytes(zip_bytes: bytes, ConditionClass):
    groups = parse_zip_group_map(zip_bytes)
    conditions = []
    for group_name, file_list in groups.items():
        cond = ConditionClass(name=group_name, description=f"Imported from ZIP group '{group_name}'")
        cond.files = []
        for fname, data in file_list:
            cond.files.append({
                "name": fname,
                "type": _infer_mime(fname),
                "size": len(data),
                "data": data
            })
        conditions.append(cond)
    return conditions
