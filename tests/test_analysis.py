import unittest
import numpy as np
import pandas as pd
from analysis import analyze_diffusion, analyze_diffusion_population
from ornstein_uhlenbeck_analyzer import analyze_ornstein_uhlenbeck

class TestAnalysis(unittest.TestCase):

    def setUp(self):
        # Create a simple tracks dataframe with one population
        self.tracks_df_one_pop = pd.DataFrame({
            'track_id': [0]*10 + [1]*10,
            'frame': list(range(10)) + list(range(10)),
            'x': list(np.linspace(0, 1, 10)) + list(np.linspace(0, 1.1, 10)),
            'y': list(np.linspace(0, 0, 10)) + list(np.linspace(0, 0, 10))
        })

        # Create a tracks dataframe with two distinct populations
        self.tracks_df_two_pops = pd.DataFrame({
            'track_id': [0]*20 + [1]*20 + [2]*20 + [3]*20,
            'frame': list(range(20)) * 4,
            'x': list(np.linspace(0, 1, 20)) + list(np.linspace(0, 1.1, 20)) + list(np.linspace(0, 10, 20)) + list(np.linspace(0, 10.1, 20)),
            'y': [0]*80
        })
        self.pixel_size = 0.1
        self.frame_interval = 0.1

    def test_analyze_diffusion_gof_and_ci(self):
        """Test that analyze_diffusion returns goodness-of-fit and CI metrics."""
        results = analyze_diffusion(self.tracks_df_one_pop, pixel_size=self.pixel_size, frame_interval=self.frame_interval)
        self.assertTrue(results['success'])
        track_results = results['track_results']
        self.assertIn('r_squared', track_results.columns)
        self.assertIn('aic', track_results.columns)
        self.assertIn('bic', track_results.columns)
        self.assertIn('diffusion_coefficient_ci_lower', track_results.columns)
        self.assertIn('diffusion_coefficient_ci_upper', track_results.columns)
        self.assertIn('alpha_ci_lower', track_results.columns)
        self.assertIn('alpha_ci_upper', track_results.columns)


    def test_analyze_diffusion_population_one_pop(self):
        """Test analyze_diffusion_population with one population."""
        results = analyze_diffusion_population(self.tracks_df_one_pop, pixel_size=self.pixel_size, frame_interval=self.frame_interval, n_populations=2)
        self.assertTrue(results['success'])
        self.assertEqual(results['n_populations'], 1)

    def test_analyze_diffusion_population_two_pops_gof(self):
        """Test analyze_diffusion_population with two populations returns GOF."""
        results = analyze_diffusion_population(self.tracks_df_two_pops, pixel_size=self.pixel_size, frame_interval=self.frame_interval, n_populations=2)
        self.assertTrue(results['success'])
        self.assertIn('gmm_bic', results)
        self.assertIn('gmm_aic', results)
        self.assertIn('gmm_n_components', results)
        self.assertGreater(results['gmm_n_components'], 1)

    def test_analyze_ornstein_uhlenbeck(self):
        """Test the Ornstein-Uhlenbeck analysis."""
        results = analyze_ornstein_uhlenbeck(self.tracks_df_one_pop, pixel_size=self.pixel_size, frame_interval=self.frame_interval)
        self.assertTrue(results['success'])
        self.assertIn('vacf_data', results)
        self.assertIn('ou_parameters', results)
        self.assertFalse(results['ou_parameters'].empty)


if __name__ == '__main__':
    unittest.main()
