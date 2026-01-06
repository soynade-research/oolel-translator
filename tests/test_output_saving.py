from unittest.mock import MagicMock, patch

from src.inference import SyntheticDataGenerator, InferRequest


def test_save_results_local(mock_vllm_engine, mock_dataset_class, base_args):
    base_args.output = "results/out.jsonl"

    generator = SyntheticDataGenerator(base_args)

    requests = [InferRequest(messages=[{"role": "user", "content": "input1"}])]

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "output1"
    responses = [mock_resp]

    mock_ds_instance = MagicMock()
    mock_dataset_class.from_list.return_value = mock_ds_instance

    with patch("pathlib.Path.mkdir"):
        generator.save_results(requests, responses)

        mock_dataset_class.from_list.assert_called_once()
        data_arg = mock_dataset_class.from_list.call_args[0][0]
        assert len(data_arg) == 1
        assert data_arg[0]["input"] == "input1"
        assert data_arg[0]["output"] == "output1"

        mock_ds_instance.to_json.assert_called_with(
            "results/out.jsonl",
            lines=True,
            force_ascii=False,
        )
        mock_ds_instance.push_to_hub.assert_not_called()


def test_save_results_hub(mock_vllm_engine, mock_dataset_class, base_args):
    base_args.output = "user/repo"
    base_args.hf_token = "fake_token"

    generator = SyntheticDataGenerator(base_args)

    requests = [InferRequest(messages=[{"role": "user", "content": "input1"}])]

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "output1"
    responses = [mock_resp]

    mock_ds_instance = MagicMock()
    mock_dataset_class.from_list.return_value = mock_ds_instance

    generator.save_results(requests, responses)

    mock_ds_instance.push_to_hub.assert_called_with("user/repo", token="fake_token")
    mock_ds_instance.to_json.assert_not_called()
