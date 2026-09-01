from unittest.mock import MagicMock, patch

import pytest

from src.inference import SyntheticDataGenerator


def test_generate_builds_config_and_passes_metrics(
    mock_vllm_engine,
    base_args,
):
    base_args.max_tokens = 77
    base_args.temperature = 0.15
    generator = SyntheticDataGenerator(base_args)
    requests = [object(), object()]
    responses = [object(), object()]
    generator.engine.infer.return_value = responses

    with (
        patch("src.inference.RequestConfig") as config_class,
        patch("src.inference.InferStats") as stats_class,
    ):
        result = generator.generate(requests)

    config_class.assert_called_once_with(max_tokens=77, temperature=0.15)
    generator.engine.infer.assert_called_once_with(
        requests,
        config_class.return_value,
        metrics=[stats_class.return_value],
    )
    stats_class.return_value.compute.assert_called_once_with()
    assert result is responses


def test_run_passes_each_stage_output_to_the_next(mock_vllm_engine, base_args):
    generator = SyntheticDataGenerator(base_args)
    requests = [object()]
    responses = [object()]
    generator.load_input_data = MagicMock(return_value=requests)
    generator.generate = MagicMock(return_value=responses)
    generator.save_results = MagicMock()

    generator.run()

    generator.generate.assert_called_once_with(requests)
    generator.save_results.assert_called_once_with(requests, responses)


def test_run_reraises_and_stops_after_generation_failure(
    mock_vllm_engine,
    base_args,
):
    generator = SyntheticDataGenerator(base_args)
    requests = [object()]
    generator.load_input_data = MagicMock(return_value=requests)
    generator.generate = MagicMock(side_effect=RuntimeError("inference failed"))
    generator.save_results = MagicMock()

    with pytest.raises(RuntimeError, match="inference failed"):
        generator.run()

    generator.save_results.assert_not_called()
