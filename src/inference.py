import os
import json
import argparse
import logging
from pathlib import Path
from typing import List

from datasets import load_dataset, Dataset
from swift.llm import (
    InferEngine,
    InferRequest,
    PtEngine,
    VllmEngine,
    RequestConfig,
)
from swift.plugin import InferStats

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Batch Inference for Synthetic Data Generation",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID or local model path",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="qwen2_5",
        help="MS-Swift model type (e.g., qwen2_5, llama3)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "pt"],
        help="Inference backend: vllm (fast) or pt (PyTorch)",
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input: local file (.json/.jsonl/.csv) or HF dataset ID",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column containing text to process",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split (for HF datasets)"
    )

    parser.add_argument(
        "--use_hf",
        type=bool,
        default=True,
        help="Use Hugging face instead of Model Scope",
    )

    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt for generation",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3, help="Sampling temperature"
    )

    # Backend-specific
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size (PyTorch backend only)"
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Context window size (vLLM backend)",
    )

    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="Fraction of each GPU memory vLLM can use (0-1).",
    )

    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM (number of GPUs to shard the model over).",
    )

    parser.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=1,
        help="Number of nodes used for pipeline parallelism",
    )

    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=256,
        help=(
            "Max number of concurrent sequences vLLM processes "
            "(controls KV cache usage)"
        ),
    )

    parser.add_argument(
        "--vllm_extra",
        type=str,
        default="{}",
        help="JSON string of extra keyword args passed directly to VllmEngine.",
    )

    # Output configuration
    parser.add_argument(
        "--output",
        type=str,
        default="generated_output.jsonl",
        help="Output path: local file path or HF repo ID (user/repo-name)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token (uses HF_TOKEN env var if not provided)",
    )

    return parser.parse_args()


class SyntheticDataGenerator:
    """
    Synthetic data generator using MS-Swift.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.engine = self._create_engine()
        self._setup_hf_token()

    def _setup_hf_token(self):
        """Configure HuggingFace token."""
        if self.args.hf_token:
            os.environ["HF_TOKEN"] = self.args.hf_token

    def _create_engine(self) -> InferEngine:
        """Initialize the inference engine based on backend."""
        logger.info(f"Initializing {self.args.backend} engine: {self.args.model}")

        common_kwargs = {
            "model_id_or_path": self.args.model,
            "model_type": self.args.model_type,
            "use_hf": self.args.use_hf,
        }

        if self.args.backend == "vllm":
            try:
                vllm_extra = json.loads(self.args.vllm_extra or "{}")
            except Exception as e:
                raise ValueError(f"Invalid JSON for --vllm-extra: {e}")

            vllm_kwargs = {
                "gpu_memory_utilization": self.args.gpu_memory_utilization,
                "tensor_parallel_size": self.args.tensor_parallel_size,
                "pipeline_parallel_size": self.args.pipeline_parallel_size,
                "max_num_seqs": self.args.max_num_seqs,
                **vllm_extra,
            }
            return VllmEngine(
                **common_kwargs,
                max_model_len=self.args.max_model_len,
                **vllm_kwargs,
            )
        else:  # pt
            return PtEngine(
                **common_kwargs,
                max_batch_size=self.args.batch_size,
            )

    def load_input_data(self) -> List[InferRequest]:
        """
        Load input dataset and prepare inference requests.
        """
        logger.info(f"Loading input data: {self.args.input}")

        # Load dataset - handles local files and HF Hub automatically
        dataset = self._load_dataset()

        logger.info(f"Loaded {len(dataset)} examples")

        # Build inference requests
        requests = [
            InferRequest(
                messages=[
                    {"role": "system", "content": self.args.system_prompt},
                    {"role": "user", "content": row[self.args.text_column]},
                ]
            )
            for row in dataset
            if self.args.text_column in row and row[self.args.text_column]
        ]

        if not requests:
            raise ValueError(
                f"No valid data found. Check that '{self.args.text_column}' "
                f"column exists and contains text."
            )

        logger.info(f"Prepared {len(requests)} inference requests")
        return requests

    def _load_dataset(self):
        """Load dataset from file or HuggingFace Hub."""
        input_path = self.args.input

        # Determine input type and load accordingly
        if input_path.endswith((".json", ".jsonl")):
            return load_dataset("json", data_files=input_path, split="train")
        elif input_path.endswith(".csv"):
            return load_dataset("csv", data_files=input_path, split="train")
        else:
            # Assume HuggingFace Hub dataset ID
            return load_dataset(input_path, split=self.args.split)

    def generate(self, requests: List[InferRequest]) -> List:
        """
        Run batch inference on requests.

        Args:
            requests: List of inference requests

        Returns:
            List of response objects
        """
        config = RequestConfig(
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
        )

        stats = InferStats()
        logger.info(f"Generating outputs for {len(requests)} examples...")

        # Prepare inference kwargs
        infer_kwargs = {"metrics": [stats]}
        # if self.args.backend == "pt":
        # 	infer_kwargs["max_batch_size"] = self.args.batch_size

        responses = self.engine.infer(requests, config, **infer_kwargs)

        logger.info(f"Generation complete. Throughput: {stats.compute()}")
        return responses

    def save_results(self, requests: List[InferRequest], responses: List):
        """
        Save results locally or push to HuggingFace Hub.

        Args:
            requests: Original inference requests
            responses: Model responses
        """
        output_data = [
            {
                "system_prompt": self.args.system_prompt,
                "input": req.messages[-1]["content"],
                "output": resp.choices[0].message.content,
            }
            for req, resp in zip(requests, responses)
        ]

        dataset = Dataset.from_list(output_data)
        output = self.args.output

        # Determine if output is HF Hub repo or local path
        is_hub_repo = "/" in output and not any(
            output.endswith(ext) for ext in [".json", ".jsonl", ".csv"]
        )

        if is_hub_repo:
            self._push_to_hub(dataset, output)
        else:
            self._save_locally(dataset, output)

    def _push_to_hub(self, dataset: Dataset, repo_id: str):
        """Push dataset to HuggingFace Hub."""
        logger.info(f"Pushing to HuggingFace Hub: {repo_id}")

        token = self.args.hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError(
                "HuggingFace token required. Provide via --hf-token or HF_TOKEN env var"
            )

        dataset.push_to_hub(repo_id, token=token)
        logger.info(
            f"Successfully pushed to: https://huggingface.co/datasets/{repo_id}"
        )

    def _save_locally(self, dataset: Dataset, output_path: str):
        """Save dataset to local file."""
        logger.info(f"Saving to local file: {output_path}")

        # Create parent directories if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save as JSONL (handles .json and .jsonl extensions)
        dataset.to_json(output_path, lines=True, force_ascii=False)
        logger.info(f"Saved {len(dataset)} examples to: {output_path}")

    def run(self):
        """Execute the complete generation pipeline."""
        try:
            # Load data
            requests = self.load_input_data()

            # Generate outputs
            responses = self.generate(requests)

            # Save results
            self.save_results(requests, responses)

            logger.info("Pipeline completed successfully!")

        except Exception as e:
            logger.error(f"✗ Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point."""
    args = parse_arguments()

    logger.info("Synthetic Data Generation Pipeline")
    logger.info(f"Model: {args.model}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    generator = SyntheticDataGenerator(args)
    generator.run()


if __name__ == "__main__":
    main()
