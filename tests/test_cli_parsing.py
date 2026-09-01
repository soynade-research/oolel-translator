import sys
from unittest.mock import patch

import pytest

from src.inference import parse_arguments


def test_parse_arguments_defaults():
    with patch.object(
        sys, "argv", ["prog", "--model", "oolel", "--input", "synth-data"]
    ):
        args = parse_arguments()
        assert args.model == "oolel"
        assert args.input == "synth-data"
        assert args.model_type == "qwen2_5"
        assert args.backend == "vllm"
        assert args.output == "generated_output.jsonl"
        assert args.temperature == 0.3


def test_parse_arguments_custom():
    with patch.object(
        sys,
        "argv",
        [
            "prog",
            "--model",
            "oolel",
            "--input",
            "synth-data",
            "--backend",
            "pt",
            "--temperature",
            "0.7",
            "--text_column",
            "source",
            "--max_tokens",
            "512",
        ],
    ):
        args = parse_arguments()
        assert args.backend == "pt"
        assert args.temperature == 0.7
        assert args.text_column == "source"
        assert args.max_tokens == 512


def test_parse_arguments_rejects_invalid_backend():
    argv = [
        "prog",
        "--model",
        "oolel",
        "--input",
        "data.json",
        "--backend",
        "invalid",
    ]
    with patch.object(sys, "argv", argv), pytest.raises(SystemExit):
        parse_arguments()
