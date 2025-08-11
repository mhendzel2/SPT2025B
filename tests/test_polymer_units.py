import pandas as pd
from biophysical_models import PolymerPhysicsModel

def test_no_double_time_scaling():
    msd = pd.DataFrame({"track_id":[1,1,1], "lag_time":[0.1,0.2,0.4], "msd":[0.02,0.03,0.05]})
    ppm = PolymerPhysicsModel(msd, pixel_size=1.0, frame_interval=0.1, lag_units="seconds")
    out = ppm.fit_rouse_model(fit_alpha=False)
    assert out["success"]
    assert "parameters" in out
