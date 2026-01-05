def test_load_input_data_json(mock_datasets, mock_vllm_engine, base_args):
    from src.inference import SyntheticDataGenerator

    base_args.input = "data.json"
    base_args.text_column = "text"

    mock_datasets.return_value = [{"text": "hello"}, {"text": "world"}]

    generator = SyntheticDataGenerator(base_args)
    requests = generator.load_input_data()

    assert len(requests) == 2
    assert requests[0].messages[1]["content"] == "hello"
    assert requests[1].messages[1]["content"] == "world"
    mock_datasets.assert_called_with("json", data_files="data.json", split="train")


def test_load_input_data_hf(mock_datasets, mock_vllm_engine, base_args):
    from src.inference import SyntheticDataGenerator

    base_args.input = "user/repo"
    base_args.split = "validation"
    base_args.text_column = "input"

    mock_datasets.return_value = [{"input": "test"}]

    generator = SyntheticDataGenerator(base_args)
    requests = generator.load_input_data()

    assert len(requests) == 1
    assert requests[0].messages[1]["content"] == "test"
    mock_datasets.assert_called_with("user/repo", split="validation")
