from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.inference import SyntheticDataGenerator, InferRequest


def _response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_save_results_local(
    tmp_path,
    mock_vllm_engine,
    mock_dataset_class,
    base_args,
):
    output_path = tmp_path / "results" / "out.jsonl"
    base_args.output = str(output_path)

    generator = SyntheticDataGenerator(base_args)

    requests = [
        InferRequest(messages=[{"role": "user", "content": "input1"}]),
        InferRequest(messages=[{"role": "user", "content": "input2"}]),
    ]
    responses = [_response("output1"), _response("output2")]

    mock_ds_instance = MagicMock()
    mock_dataset_class.from_list.return_value = mock_ds_instance

    generator.save_results(requests, responses)

    mock_dataset_class.from_list.assert_called_once_with(
        [
            {"system_prompt": "sys", "input": "input1", "output": "output1"},
            {"system_prompt": "sys", "input": "input2", "output": "output2"},
        ]
    )
    assert output_path.parent.is_dir()

    mock_ds_instance.to_json.assert_called_once_with(
        str(output_path),
        lines=True,
        force_ascii=False,
    )
    mock_ds_instance.push_to_hub.assert_not_called()


def test_save_results_hub(
    monkeypatch,
    mock_vllm_engine,
    mock_dataset_class,
    base_args,
):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    base_args.output = "user/repo"
    base_args.hf_token = "fake_token"

    generator = SyntheticDataGenerator(base_args)

    requests = [InferRequest(messages=[{"role": "user", "content": "input1"}])]

    responses = [_response("output1")]

    mock_ds_instance = MagicMock()
    mock_dataset_class.from_list.return_value = mock_ds_instance

    generator.save_results(requests, responses)

    mock_dataset_class.from_list.assert_called_once_with(
        [{"system_prompt": "sys", "input": "input1", "output": "output1"}]
    )
    mock_ds_instance.push_to_hub.assert_called_once_with(
        "user/repo",
        token="fake_token",
    )
    mock_ds_instance.to_json.assert_not_called()


def test_hub_save_requires_token(
    monkeypatch,
    mock_vllm_engine,
    base_args,
):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    generator = SyntheticDataGenerator(base_args)
    dataset = MagicMock()

    with pytest.raises(ValueError, match="HuggingFace token required"):
        generator._push_to_hub(dataset, "user/repo")

    dataset.push_to_hub.assert_not_called()
