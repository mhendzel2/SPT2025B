from __future__ import annotations
import io
import pandas as pd
from typing import Dict, List

CSV_CANDIDATES = [
    ('track_id','frame','x','y'),
    ('TRACK_ID','FRAME','POSITION_X','POSITION_Y'),
    ('TrackID','Frame','X','Y'),
]

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower(): c for c in df.columns}
    if {'track_id','frame','x','y'}.issubset(lower):
        return pd.DataFrame({
            'track_id': df[lower['track_id']],
            'frame': df[lower['frame']],
            'x': df[lower['x']],
            'y': df[lower['y']],
        })
    for tid, frm, xx, yy in CSV_CANDIDATES[1:]:
        if {tid, frm, xx, yy}.issubset(df.columns):
            return pd.DataFrame({
                'track_id': df[tid],
                'frame': df[frm],
                'x': df[xx],
                'y': df[yy],
            })
    raise ValueError("Unrecognized column set for track data")

def _load_one_file(file_info: Dict, condition_name: str) -> pd.DataFrame:
    name = file_info.get('name','data')
    data = file_info.get('data', b'')
    ext = (name.split('.')[-1] or '').lower()
    if ext == 'csv':
        df = pd.read_csv(io.BytesIO(data))
    elif ext == 'xlsx':
        df = pd.read_excel(io.BytesIO(data))
    elif ext in ('xml','mvd2'):
        raise NotImplementedError(f"Parser for .{ext} not implemented")
    else:
        raise ValueError(f"Unsupported file type: .{ext}")
    out = _normalize_df(df)
    out['condition'] = condition_name
    return out

def assemble_tracks_from_project(project) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for cond in getattr(project, 'conditions', []):
        for f in getattr(cond, 'files', []):
            try:
                frames.append(_load_one_file(f, cond.name))
            except Exception as e:
                print(f"[assemble_tracks] skip {f.get('name')}: {e}")
    if not frames:
        return pd.DataFrame(columns=['track_id','frame','x','y','condition'])
    df = pd.concat(frames, ignore_index=True)
    df['track_id'] = df['track_id'].astype(int)
    df['frame'] = df['frame'].astype(int)
    return df.sort_values(['track_id','frame']).reset_index(drop=True)
