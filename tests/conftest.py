import argparse
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest


class InferRequestStub:
    def __init__(self, messages):
        self.messages = messages


def _stub_module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


# Unit tests replace all runtime collaborators; importing them must not require
# vLLM, CUDA, datasets, network access, or model downloads.
_stub_module("datasets", load_dataset=MagicMock(), Dataset=MagicMock())
llm = _stub_module(
    "swift.llm",
    InferEngine=object,
    InferRequest=InferRequestStub,
    PtEngine=MagicMock(),
    VllmEngine=MagicMock(),
    RequestConfig=MagicMock(),
)
plugin = _stub_module("swift.plugin", InferStats=MagicMock())
_stub_module("swift", llm=llm, plugin=plugin)


@pytest.fixture
def base_args(tmp_path):
    return argparse.Namespace(
        input="data.json",
        text_column="text",
        split="train",
        model="model",
        model_type="qwen2_5",
        backend="vllm",
        hf_token=None,
        use_hf=True,
        system_prompt="sys",
        max_tokens=100,
        temperature=0.3,
        batch_size=32,
        max_model_len=None,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_num_seqs=256,
        vllm_extra="{}",
        output=str(tmp_path / "out.jsonl"),
    )


@pytest.fixture
def mock_datasets(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("src.inference.load_dataset", mock)
    return mock


@pytest.fixture
def mock_vllm_engine(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("src.inference.VllmEngine", mock)
    return mock


@pytest.fixture
def mock_pt_engine(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("src.inference.PtEngine", mock)
    return mock


@pytest.fixture
def mock_dataset_class(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("src.inference.Dataset", mock)
    return mock
