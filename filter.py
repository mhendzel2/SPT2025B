import pandas as pd

def filter_tracks_by_length(tracks_df: pd.DataFrame, min_length: int) -> pd.DataFrame:
    """
    Filter tracks by length.

    Args:
        tracks_df: DataFrame of tracks with a 'track_id' column.
        min_length: The minimum length of a track to be kept.

    Returns:
        A DataFrame with only the tracks that are longer than min_length.
    """
    track_lengths = tracks_df.groupby('track_id').size()
    valid_tracks = track_lengths[track_lengths >= min_length].index
    return tracks_df[tracks_df['track_id'].isin(valid_tracks)].copy()
