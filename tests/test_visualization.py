import unittest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from hmm_analysis import fit_hmm
from visualization import plot_hmm_state_transition_diagram

class TestVisualization(unittest.TestCase):

    def setUp(self):
        # Create a simple tracks dataframe
        self.tracks_df = pd.DataFrame({
            'track_id': [0]*10 + [1]*10,
            'frame': list(range(10)) + list(range(10)),
            'x': np.concatenate([np.random.randn(10).cumsum(), np.random.randn(10).cumsum() + 10]),
            'y': np.concatenate([np.random.randn(10).cumsum(), np.random.randn(10).cumsum()])
        })

    def test_plot_hmm_state_transition_diagram(self):
        """Test that plot_hmm_state_transition_diagram returns a Plotly figure."""
        n_states = 2
        model, _ = fit_hmm(self.tracks_df, n_states=n_states, n_iter=10)
        self.assertIsNotNone(model)

        fig = plot_hmm_state_transition_diagram(model)

        # Check if the returned object is a Plotly figure
        self.assertIsInstance(fig, go.Figure)

        # Check for the presence of nodes and annotations
        # There should be one trace for the nodes
        self.assertEqual(len(fig.data), 1)
        # The number of nodes should be equal to n_states
        self.assertEqual(len(fig.data[0].x), n_states)

        # The number of annotations can vary depending on the transition probabilities,
        # but we expect at least the node labels.
        self.assertGreaterEqual(len(fig.layout.annotations), n_states)

if __name__ == '__main__':
    unittest.main()
