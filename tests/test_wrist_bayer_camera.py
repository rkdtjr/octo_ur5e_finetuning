import numpy as np
import pytest
from octo_ur5e_collector.wrist_bayer_camera_node import debayer_grbg

def test_debayer_grbg_shape_and_type():
    raw=np.arange(8*6,dtype=np.uint8).reshape(6,8)
    rgb=debayer_grbg(raw,8,6)
    assert rgb.shape==(6,8,3)
    assert rgb.dtype==np.uint8
    assert rgb.flags.c_contiguous

def test_debayer_rejects_wrong_byte_count():
    with pytest.raises(ValueError):debayer_grbg(np.zeros((3,3),np.uint8),8,6)
