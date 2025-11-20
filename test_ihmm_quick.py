"""Quick verification test for iHMM implementation."""

import numpy as np
from ihmm_analysis import BlurAwareHMM, InfiniteHMM

print("=" * 60)
print("iHMM Quick Verification Test")
print("=" * 60)

# Generate simple 2-state synthetic data
np.random.seed(42)
n_steps = 50

# State 1: Slow (D=0.01 µm²/s)
displacements_slow = np.random.randn(25, 2) * np.sqrt(2 * 0.01 * 0.1)

# State 2: Fast (D=0.5 µm²/s)
displacements_fast = np.random.randn(25, 2) * np.sqrt(2 * 0.5 * 0.1)

# Concatenate
displacements = np.vstack([displacements_slow, displacements_fast])

print(f"\n1. Generated {len(displacements)} displacements (2 states)")
print(f"   True D: [0.01, 0.5] µm²/s")

# Test fixed-state HMM
print("\n2. Testing BlurAwareHMM (2 states)...")
model = BlurAwareHMM(n_states=2, frame_interval=0.1)
result = model.fit(displacements, max_iter=50)

if result['success']:
    print(f"   ✓ Converged: {result['n_iterations']} iterations")
    print(f"   Estimated D: {model.diffusion_coefficients}")
else:
    print(f"   ✗ Failed")

# Test automatic state selection
print("\n3. Testing InfiniteHMM (auto-selection)...")
ihmm = InfiniteHMM(frame_interval=0.1, method='BIC')
result_ihmm = ihmm.fit(displacements, min_states=2, max_states=4, max_iter=50)

if result_ihmm['success']:
    print(f"   ✓ Selected: {result_ihmm['best_n_states']} states")
    print(f"   Scores: {[(k, v['score']) for k, v in result_ihmm['model_scores'].items()]}")
    print(f"   D values: {result_ihmm['diffusion_coefficients']}")
else:
    print(f"   ✗ Failed: {result_ihmm.get('error', 'Unknown')}")

print("\n" + "=" * 60)
print("Test complete!")
