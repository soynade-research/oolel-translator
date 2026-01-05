import sys
from unittest.mock import patch

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
        ],
    ):
        args = parse_arguments()
        assert args.backend == "pt"
        assert args.temperature == 0.7
