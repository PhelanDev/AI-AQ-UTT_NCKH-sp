import pytest
import numpy as np
from pi.spatial_temporal_engine import SpatialTemporalEngine

def test_idw_unified_heatmap_generation():
    engine = SpatialTemporalEngine()
    vals = [25.0, 35.0, 45.0, 55.0, 30.0, 40.0, 50.0, 60.0]
    res = engine.generate_pollution_heatmap(vals, grid_resolution=20)

    assert res["grid_resolution"] == 20
    assert res["heatmap_pm25"].shape == (20, 20)
    assert not np.isnan(res["heatmap_pm25"]).any()
    assert not np.isinf(res["heatmap_pm25"]).any()
    assert 25.0 <= res["mean_pm25"] <= 60.0
    assert res["execution_time_ms"] < 20.0  # Siêu nhanh < 20ms
