import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def exponential_decay(t, tau, A):
    """Exponential decay function."""
    return A * np.exp(-t / tau)

def calculate_vacf(tracks_df, pixel_size=1.0, frame_interval=1.0):
    """
    Calculate the velocity autocorrelation function for each track.
    """
    vacf_data = []
    for track_id, track in tracks_df.groupby('track_id'):
        if len(track) < 2:
            continue

        x = track['x'].values * pixel_size
        y = track['y'].values * pixel_size
        t = track['frame'].values * frame_interval

        vx = np.diff(x) / np.diff(t)
        vy = np.diff(y) / np.diff(t)

        vacf = []
        lags = range(len(vx) // 2)
        for lag in lags:
            if lag == 0:
                vacf.append(np.mean(vx*vx + vy*vy))
            else:
                vacf.append(np.mean(vx[:-lag]*vx[lag:] + vy[:-lag]*vy[lag:]))

        vacf_data.append(pd.DataFrame({
            'lag': np.array(lags) * frame_interval,
            'vacf': vacf,
            'track_id': track_id
        }))

    return pd.concat(vacf_data, ignore_index=True) if vacf_data else pd.DataFrame()

def fit_vacf(vacf_df):
    """
    Fit the VACF to an exponential decay to extract OU parameters.
    """
    fit_results = []
    for track_id, track_vacf in vacf_df.groupby('track_id'):
        lags = track_vacf['lag'].values
        vacf = track_vacf['vacf'].values

        try:
            # Fit the data
            popt, pcov = curve_fit(exponential_decay, lags, vacf, p0=(1.0, vacf[0]))
            tau, A = popt

            # Extract parameters
            # Assuming equipartition theorem: <v^2> = kT/m
            # And for OU process: tau = m/gamma, D = kT/gamma
            # So, D = <v^2> * tau
            # and gamma = m/tau, k = m/tau^2
            # We can't get m, so we get ratios
            kt_m = A # <v^2>
            gamma_m = 1/tau # gamma/m
            k_m = 1/tau**2 # k/m

            fit_results.append({
                'track_id': track_id,
                'relaxation_time': tau,
                'kt_m': kt_m,
                'gamma_m': gamma_m,
                'k_m': k_m,
                'fit_successful': True
            })
        except RuntimeError:
            fit_results.append({
                'track_id': track_id,
                'relaxation_time': np.nan,
                'kt_m': np.nan,
                'gamma_m': np.nan,
                'k_m': np.nan,
                'fit_successful': False
            })

    return pd.DataFrame(fit_results)

def analyze_ornstein_uhlenbeck(tracks_df, pixel_size=1.0, frame_interval=1.0):
    """
    Main function to perform Ornstein-Uhlenbeck analysis.
    """
    if tracks_df.empty:
        return {
            'success': False,
            'error': 'Input dataframe is empty.',
            'vacf_data': pd.DataFrame(),
            'ou_parameters': pd.DataFrame()
        }

    vacf_df = calculate_vacf(tracks_df, pixel_size, frame_interval)

    if vacf_df.empty:
        return {
            'success': False,
            'error': 'Could not calculate VACF.',
            'vacf_data': pd.DataFrame(),
            'ou_parameters': pd.DataFrame()
        }

    ou_parameters = fit_vacf(vacf_df)

    return {
        'success': True,
        'vacf_data': vacf_df,
        'ou_parameters': ou_parameters
    }
