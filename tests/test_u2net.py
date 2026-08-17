from types import SimpleNamespace

import numpy as np
import pytest

from openscan_eval.preprocessing.u2net import U2NetSegmenter


class FakeSession:
    def __init__(self,path,providers): self.path=path
    def get_inputs(self): return [SimpleNamespace(name="discovered_input",shape=[1,3,8,10])]
    def get_outputs(self): return [SimpleNamespace(name="discovered_output",shape=[1,1,8,10])]
    def run(self,names,feeds):
        assert names==["discovered_output"] and "discovered_input" in feeds
        output=np.zeros((1,1,8,10),np.float32);output[:,:,2:6,3:8]=.75;return [output]


def test_u2net_model_path_validation(tmp_path):
    with pytest.raises(FileNotFoundError,match="U2Net model not found"):
        U2NetSegmenter(tmp_path/"missing.onnx",{},FakeSession)


def test_onnx_adapter_discovers_metadata_and_resizes_mask(tmp_path):
    model=tmp_path/"model.onnx";model.write_bytes(b"mock")
    segmenter=U2NetSegmenter(model,{},FakeSession);image=np.zeros((24,37,3),np.uint8)
    probability=segmenter.predict(image)
    assert segmenter.input.name=="discovered_input"
    assert probability.shape==(24,37) and 0<probability.max()<=1
