import unittest
import numpy as np
import pandas as pd
from analysis import calculate_msd, analyze_diffusion, analyze_motion
import fbm

class TestAnalysisFormulas(unittest.TestCase):

    def setUp(self):
        """Set up test data."""
        # General parameters
        self.pixel_size = 1.0  # um/pixel
        self.frame_interval = 0.1  # s/frame

        # 1. Empty dataframe
        self.empty_df = pd.DataFrame({'track_id': [], 'frame': [], 'x': [], 'y': []})

        # 2. Track with a single point
        self.single_point_df = pd.DataFrame({'track_id': [0], 'frame': [0], 'x': [1], 'y': [1]})

        # 3. Track with constant velocity
        self.velocity_x = 10.0  # um/s
        const_vel_frames = np.arange(20)
        self.const_vel_df = pd.DataFrame({
            'track_id': 0,
            'frame': const_vel_frames,
            'x': self.velocity_x * const_vel_frames * self.frame_interval,
            'y': 0
        })

        # 4. Track with normal diffusion (Brownian motion)
        self.known_D = 0.5  # um^2/s
        sigma = np.sqrt(2 * self.known_D * self.frame_interval)
        n_steps = 1000
        np.random.seed(42)
        steps_x = np.random.normal(0, sigma, n_steps)
        steps_y = np.random.normal(0, sigma, n_steps)
        self.normal_diffusion_df = pd.DataFrame({
            'track_id': 0,
            'frame': np.arange(n_steps + 1),
            'x': np.cumsum(np.insert(steps_x, 0, 0)),
            'y': np.cumsum(np.insert(steps_y, 0, 0))
        })

        # 5. Track with anomalous diffusion (subdiffusion)
        self.known_alpha = 0.6
        hurst = self.known_alpha / 2.0  # H = alpha / 2
        n_steps_anomalous = 1000
        x_fbm = fbm.fbm(n=n_steps_anomalous, hurst=hurst, length=1, method='daviesharte')
        y_fbm = fbm.fbm(n=n_steps_anomalous, hurst=hurst, length=1, method='daviesharte')
        self.anomalous_diffusion_df = pd.DataFrame({
            'track_id': 0,
            'frame': np.arange(n_steps_anomalous + 1),
            'x': x_fbm,
            'y': y_fbm
        })

    def test_calculate_msd_empty_input(self):
        """Test calculate_msd with an empty dataframe."""
        msd_df = calculate_msd(self.empty_df)
        self.assertTrue(msd_df.empty)

    def test_calculate_msd_single_point_track(self):
        """Test calculate_msd with a track that is too short."""
        msd_df = calculate_msd(self.single_point_df, min_track_length=2)
        self.assertTrue(msd_df.empty)

    def test_calculate_msd_constant_velocity(self):
        """Test calculate_msd with a constant velocity track."""
        max_lag = 10
        # Use a frame interval of 1.0s for this test to match the original setup
        frame_interval_const = 1.0
        const_vel_frames = np.arange(20)
        const_vel_df = pd.DataFrame({
            'track_id': 0,
            'frame': const_vel_frames,
            'x': self.velocity_x * const_vel_frames * frame_interval_const,
            'y': 0
        })
        msd_df = calculate_msd(
            const_vel_df,
            max_lag=max_lag,
            pixel_size=self.pixel_size,
            frame_interval=frame_interval_const
        )

        self.assertFalse(msd_df.empty)

        for lag_time, group in msd_df.groupby('lag_time'):
            theoretical_msd = (self.velocity_x * lag_time)**2
            calculated_msd = group['msd'].iloc[0]
            self.assertAlmostEqual(theoretical_msd, calculated_msd, places=5, msg=f"MSD mismatch at lag {lag_time}s")

    def test_analyze_diffusion_normal(self):
        """Test analyze_diffusion with a normally diffusing particle."""
        results = analyze_diffusion(
            self.normal_diffusion_df,
            pixel_size=self.pixel_size,
            frame_interval=self.frame_interval
        )
        self.assertTrue(results['success'])

        # Check that the calculated diffusion coefficient is close to the known value.
        # It won't be exact due to the stochastic nature of the simulation.
        calculated_D = results['ensemble_results']['mean_diffusion_coefficient']
        self.assertAlmostEqual(self.known_D, calculated_D, delta=0.1)

        # Check that the anomalous exponent is close to 1.
        calculated_alpha = results['ensemble_results']['mean_alpha']
        self.assertAlmostEqual(1.0, calculated_alpha, delta=0.15)

    def test_analyze_diffusion_anomalous(self):
        """Test analyze_diffusion with an anomalously diffusing particle."""
        results = analyze_diffusion(
            self.anomalous_diffusion_df,
            pixel_size=self.pixel_size,
            frame_interval=self.frame_interval
        )
        self.assertTrue(results['success'])

        # Check that the anomalous exponent is close to the known value.
        # This can have a higher variance.
        calculated_alpha = results['ensemble_results']['mean_alpha']
        self.assertAlmostEqual(self.known_alpha, calculated_alpha, delta=0.2)

    def test_analyze_motion_constant_velocity(self):
        """Test analyze_motion for a track with constant velocity."""
        results = analyze_motion(
            self.const_vel_df,
            pixel_size=self.pixel_size,
            frame_interval=self.frame_interval
        )
        self.assertTrue(results['success'])

        # Check ensemble results
        self.assertAlmostEqual(results['ensemble_results']['mean_speed'], self.velocity_x, places=5)

        # Check track-specific results
        track_results = results['track_results']
        self.assertEqual(len(track_results), 1)

        # Speed
        self.assertAlmostEqual(track_results['mean_speed'].iloc[0], self.velocity_x, places=5)

        # Straightness should be 1.0 for a perfectly straight line
        self.assertAlmostEqual(track_results['straightness'].iloc[0], 1.0, places=5)

        # Angle changes should be zero
        self.assertAlmostEqual(track_results['mean_abs_angle_change'].iloc[0], 0.0, places=5)


if __name__ == '__main__':
    unittest.main()
