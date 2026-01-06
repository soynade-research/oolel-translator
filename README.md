# Synthetic Data Generation

Batch inference pipeline for generating synthetic translation data using MS-Swift with vLLM backend.

## Features

- Fast batch inference with vLLM (primary) and PyTorch (fallback)
- Automatic dataset loading from HuggingFace Hub or local files
- Support for multiple input formats (JSONL, JSON, CSV)
- Configurable system prompts and generation parameters
- Performance metrics and throughput tracking
- Save locally or push directly to HuggingFace Hub

## Installation

First, clone the repository and navigate into the directory:

```bash
git clone https://github.com/soynade-research/synthetic-data-generation.git
cd synthetic-data-generation
```


### 1. Option 1 (Recommanded)

You can automate the installation of `uv`, the virtual environment creation, and dependency installation using the provided script.

```bash
source setup.sh
```

Using source to run the script ensures the virtual environment is automatically activated in your current terminal.

**Note:** If you run `bash setup.sh` instead, the installation will succeed, but you will need to manually run `source .venv/bin/activate` afterwards.


### 2. Manual Installation

If you prefer to install the dependencies step-by-step, follow the instructions below.
1. **Install UV**: 

If you don't have uv installed, get it via curl:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Create and Activate Virtual Environment**
```bash
# Create a virtual environment with Python 3.11
uv venv --python 3.11

# Activate the environment
source .venv/bin/activate
```
3. **Install dependancies**

```bash

uv pip install vllm --torch-backend=auto
uv pip install 'ms-swift[llm]'
```

### 3. Configure HuggingFace Token
If you wish to push the generated data in the Hub, you can configure your access token by running below command on your terminal (you can also pass directly the access token in `run.sh`)

```bash
export HF_TOKEN="ACCESS_TOKEN"
```

## Quick Start

### Basic Usage (Save Locally)

```bash
python src/inference.py \
    --model "soynade-research/Oolel-Small-v0.1" \
    --model_type "qwen2_5" \
    --input "data/sample_input.jsonl" \
    --split "train" \
    --text_column "input" \
    --system_prompt "Translate to Wolof the following sentence"
```

### Push to HuggingFace Hub

```bash
HF_TOKEN=TOKEN python src/inference.py \
    --model "soynade-research/Oolel-Small-v0.1" \
    --model_type "qwen2_5" \
    --input "hf_user_name/dataset_name" \
    --split "train" \
    --text_column "input" \
    --output "user_name/output_dataset" \
    --system_prompt "Translate to Wolof the following sentence"
```

## Usage Examples

### 1. Local File → Local File

```bash
python inference_pipeline.py \
    --model "soynade-research/Oolel-Small-v0.1" \
    --input "data/sample_input.jsonl" \
    --text-column "text" \
    --output "output/wolof_translations.jsonl" \
    --system-prompt "Translate to Wolof:" \
    --temperature 0.1
```

### 2. HuggingFace Dataset → Local File

```bash
python src/inference.py \
    --model "soynade-research/Oolel-Small-v0.1" \
    --model_type "qwen2_5" \
    --input "data/input.jsonl" \
    --split "train" \
    --text_column "input" \
    --output "wolof_translations.jsonl" \
    --system_prompt "Translate to Wolof the following sentence"
```

### 3. Local File → HuggingFace Hub

```bash
HF_TOKEN=TOKEN python src/inference.py \
    --model "soynade-research/Oolel-Small-v0.1" \
    --model_type "qwen2_5" \
    --input "data.jsonl" \
    --split "train" \
    --text_column "input" \
    --output "user_name/output_dataset" \
    --system_prompt "Translate to Wolof the following sentence"
```

### 4. HuggingFace Dataset → HuggingFace Hub

```bash
HF_TOKEN=TOKEN python src/inference.py \
    --model "soynade-research/Oolel-Small-v0.1" \
    --model_type "qwen2_5" \
    --input "hf_user_name/dataset_name" \
    --split "train" \
    --text_column "input" \
    --output "user_name/output_dataset" \
    --system_prompt "Translate to Wolof the following sentence"
```

### 5. Using PyTorch Backend

```bash
python src/inference.py \
    --model "soynade-research/Oolel-Small-v0.1"\
    --backend pt \
    --model_type qwen2_5 \
    --text_column "input" \
    --batch_size 4 \
    --input "data/input.jsonl" \
    --output "output/synthetic.jsonl"
```

## Command Line Options

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--model` | HuggingFace model ID or local path |
| `--input` | Input: local file (.json/.jsonl/.csv) or HF dataset ID |
| `--text_column` | Column name containing text to process |

### Optional Arguments

| Argument | Default | Description |
|--------------|-------------|-----------------|
| `--model_type` | qwen2_5 | MS-Swift model type (e.g., qwen2_5, llama3) |
| `--backend` | vllm | Inference backend: `vllm` or `pt`|
| `--output` | generated_output.jsonl | Output: local path or HF repo ID |
| `--split` | train | Dataset split (for HF datasets) |
| `--system_prompt` | "You are a helpful assistant." | System prompt |
| `--max_tokens` | 1024 | Maximum tokens to generate |
| `--temperature` | 0.3 | Sampling temperature (0.0-1.0) |
| `--batch_size` | 32 | Batch size (PyTorch backend only) |
| `--max_model_len` | None | Context window (vLLM backend) |
| `--gpu_memory_utilization` | 0.9 | Fraction of each GPU memory vLLM can use (0-1) |
| `--tensor_parallel_size` | 0.3 | Number of GPUs to shard the model over |
| `--pipeline_parallel_size` | 32 | Number of nodes used for pipeline parallelism |
| `--max_num_seqs` | 256 | Max number of concurrent sequences vLLM processes|
| `--vllm_extra` | "{}" | SON string of extra keyword args passed directly to VllmEngine |
| `--hf_token` | None | HuggingFace token (or use HF_TOKEN env var) |


## Input Format

### JSONL (Recommended)

```jsonl
{"text": "Hello, how are you?", "id": 1}
{"text": "The weather is nice today.", "id": 2}
```

### JSON

```json
[
    {"text": "Hello, how are you?"},
    {"text": "The weather is nice today."}
]
```

### CSV

```csv
id,text
1,"Hello, how are you?"
2,"The weather is nice today."
```

## Output Format

Results are saved in JSONL format with three fields:

```jsonl
{"system_prompt": "Translate to Wolof:", "input": "english or french input", "output": "model translation"}
{"system_prompt": "Translate to Wolof:", "input": "english or french input", "output": "model translation"}
```

## How Output Works

The `--output` parameter handles both local files and HuggingFace Hub:

### Local File (Default)
If output path ends with `.json`, `.jsonl`, or `.csv`, or doesn't contain `/`:
```bash
--output "translations.jsonl"           # Saves locally
--output "outputs/results.jsonl"        # Saves locally with directory
```

### HuggingFace Hub
If output contains `/` and doesn't end with file extension:
```bash
--output "username/dataset-name"        # Pushes to Hub
--output "org-name/translated-data"     # Pushes to Hub
```

## Performance Tips

1. **Use vLLM**: faster than PyTorch
   ```bash
   --backend vllm  # Default
   ```

2. **Adjust batch size** (PyTorch only):
   ```bash
   --backend pt --batch_size 16
   ```

3. **Lower temperature** for consistent translations:
   ```bash
   --temperature 0.0  
   ```

4. **Set context window** for long texts (vLLM):
   ```bash
   --max_model_len 8192
   ```

## Troubleshooting

### Out of Memory

Reduce batch size or max model length. For instance:
```bash
--batch_size 8                # For PyTorch
--max_model_len 1024         # For vLLM
```

### Missing Column Error

Check your column name:
```bash
--text_column "english"  # Use correct column name
```

### HuggingFace Token Error

Set token via environment variable:
```bash
export HF_TOKEN="your_token_here"
```

Or pass directly:
```bash
--hf_token "your_token_here"
```

## Project Structure

```
synthetic-data-generation/
├── inference_pipeline.py    # Main script
├── README.md               # This file
├── pyproject.toml         # UV dependencies
├── setup.sh               # Setup script
├── .env.example           # Environment template
├── .gitignore
├── data/
│   └── sample_input.jsonl
└── outputs/
    └── .gitkeep
```

## Environment Variables

Create a `.env` file:

```bash
HF_TOKEN=your_huggingface_token
CUDA_VISIBLE_DEVICES=0
```

## Advanced Usage

### Custom Model Types

```bash
python inference_pipeline.py \
    --model "path/to/model" \
    --model-type "qwen2_5" \
    --input "data/input.jsonl" \
    --output "output.jsonl"
```

### Multi-GPU (vLLM with tensor parallelism)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python inference_pipeline.py \
    --model "large-model" \
    --backend vllm \
    --input "data/input.jsonl" \
    --output "output.jsonl"
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

```bibtex
@software{synthetic_data_generation,
  title={Synthetic Data Generation},
  author={Soynade Research},
  year={2025},
  url={https://github.com/soynade-research/synthetic-data-generation}
}
```