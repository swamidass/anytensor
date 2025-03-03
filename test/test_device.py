import torch as th
import anytensor as at
import pytest

# check if mps available on the device
not_mps: bool = not th.backends.mps.is_available()

@pytest.mark.skipif(not_mps, reason="MPS is not available")
def test_device_check():
    r = th.arange(10, device="mps:0")
    assert r.device == th.device("mps:0")
