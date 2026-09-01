import pytest

from src.inference import SyntheticDataGenerator


@pytest.mark.parametrize(
    ("filename", "loader"),
    [("data.json", "json"), ("data.jsonl", "json"), ("data.csv", "csv")],
)
def test_loads_local_formats_and_builds_prompts(
    filename,
    loader,
    mock_datasets,
    mock_vllm_engine,
    base_args,
):
    base_args.input = filename
    base_args.system_prompt = "Translate to Wolof."
    mock_datasets.return_value = [
        {"text": "hello"},
        {"text": ""},
        {"other": "missing"},
        {"text": "world"},
    ]

    generator = SyntheticDataGenerator(base_args)
    requests = generator.load_input_data()

    assert [request.messages for request in requests] == [
        [
            {"role": "system", "content": "Translate to Wolof."},
            {"role": "user", "content": text},
        ]
        for text in ("hello", "world")
    ]
    mock_datasets.assert_called_once_with(
        loader,
        data_files=filename,
        split="train",
    )


def test_load_input_data_hf(mock_datasets, mock_vllm_engine, base_args):
    base_args.input = "user/repo"
    base_args.split = "validation"
    base_args.text_column = "input"

    mock_datasets.return_value = [{"input": "test"}]

    generator = SyntheticDataGenerator(base_args)
    requests = generator.load_input_data()

    assert len(requests) == 1
    assert requests[0].messages[1]["content"] == "test"
    mock_datasets.assert_called_once_with("user/repo", split="validation")


def test_rejects_input_without_usable_text(
    mock_datasets,
    mock_vllm_engine,
    base_args,
):
    mock_datasets.return_value = [{"text": ""}, {"other": "missing"}]
    generator = SyntheticDataGenerator(base_args)

    with pytest.raises(ValueError, match="'text'.*contains text"):
        generator.load_input_data()
