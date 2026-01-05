import pytest

from src.inference import SyntheticDataGenerator


def test_vllm_args_wiring(mock_vllm_engine, base_args):
    base_args.backend = "vllm"
    base_args.max_model_len = 4096
    base_args.gpu_memory_utilization = 0.5
    base_args.tensor_parallel_size = 2
    base_args.pipeline_parallel_size = 1
    base_args.max_num_seqs = 128
    base_args.vllm_extra = '{"enforce_eager": true}'

    SyntheticDataGenerator(base_args)  # ctor calls _create_engine

    mock_vllm_engine.assert_called_once()
    call_kwargs = mock_vllm_engine.call_args.kwargs

    assert call_kwargs["model_id_or_path"] == "model"
    assert call_kwargs["model_type"] == "qwen2_5"
    assert call_kwargs["gpu_memory_utilization"] == 0.5
    assert call_kwargs["tensor_parallel_size"] == 2
    assert call_kwargs["max_model_len"] == 4096
    assert call_kwargs["max_num_seqs"] == 128
    assert call_kwargs["enforce_eager"] is True


def test_pt_args_wiring(mock_pt_engine, base_args):
    base_args.backend = "pt"
    base_args.model_type = "llama3"
    base_args.batch_size = 16

    SyntheticDataGenerator(base_args)

    mock_pt_engine.assert_called_once()
    call_kwargs = mock_pt_engine.call_args.kwargs

    assert call_kwargs["model_id_or_path"] == "model"
    assert call_kwargs["model_type"] == "llama3"
    assert call_kwargs["max_batch_size"] == 16


def test_vllm_extra_invalid_json_raises(mock_vllm_engine, base_args):
    base_args.backend = "vllm"
    base_args.vllm_extra = "{invalid json"

    with pytest.raises(ValueError):
        SyntheticDataGenerator(base_args)
