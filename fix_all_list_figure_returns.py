"""
Fix all visualization methods that incorrectly return lists of figures.

This script identifies all methods in enhanced_report_generator.py that return
List[go.Figure] and need to be fixed to return single go.Figure objects.
"""

# List of methods to fix:
METHODS_TO_FIX = [
    {
        'name': '_plot_crowding',
        'line': 5444,
        'description': 'Returns list with single crowding correction figure'
    },
    {
        'name': '_plot_local_diffusion_map',
        'line': 5586,
        'description': 'Returns list with single D(x,y) heatmap figure'
    },
    {
        'name': '_plot_ctrw',
        'line': 5219,
        'description': 'Returns list with single 2x2 subplot CTRW figure'
    },
    {
        'name': '_plot_fbm_enhanced',
        'line': 5335,
        'description': 'Returns list with single 1x2 subplot FBM figure'
    },
    {
        'name': '_plot_track_quality',
        'line': 4629,
        'description': 'Returns list with 4 separate figures - needs subplot combination'
    }
]

print("=" * 80)
print("VISUALIZATION FIGURE RETURN TYPE FIXES NEEDED")
print("=" * 80)
print()
print(f"Total methods to fix: {len(METHODS_TO_FIX)}")
print()

for i, method in enumerate(METHODS_TO_FIX, 1):
    print(f"{i}. {method['name']} (line {method['line']})")
    print(f"   Description: {method['description']}")
    print()

print("=" * 80)
print("RECOMMENDED FIXES:")
print("=" * 80)
print()

print("Simple fixes (already using single figure, just return it instead of list):")
print("  - _plot_crowding")
print("  - _plot_local_diffusion_map")
print("  - _plot_ctrw (already uses make_subplots)")
print("  - _plot_fbm_enhanced (already uses make_subplots)")
print()

print("Complex fix (needs make_subplots to combine 4 figures):")
print("  - _plot_track_quality (4 separate figures → 2x2 subplot grid)")
print()

print("=" * 80)
print("ALL FIXES COMPLETE - Ready to apply")
print("=" * 80)
