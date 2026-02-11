# Oolel Translator

A batch inference pipeline designed for generating synthetic translation datasets. It utilizes MS-Swift with a vLLM backend to perform high-throughput translation tasks, supporting both local storage and direct HuggingFace Hub integration.

## Features

* **Batch Inference:** Optimized for speed using vLLM (primary) with a PyTorch fallback.
* **Flexible Input/Output:** Supports JSONL, JSON, and CSV formats.
* **Hub Integration:** Automatically loads datasets from and pushes results to the HuggingFace Hub.
* **Customizable Generation:** Configurable system prompts, temperature, and context windows.
* **Resource Management:** Options for tensor parallelism and GPU memory utilization.

## Installation

First, clone the repository and navigate into the directory:

```bash
git clone https://github.com/soynade-research/oolel-translator.git
cd oolel-translator
```


### 1. Option 1: Automated Setup (Recommended)
Run the setup script to install `uv`, create the virtual environment, and install dependencies.

```bash
source setup.sh
```

Using source to run the script ensures the virtual environment is automatically activated in your current terminal.

If you run `bash setup.sh` instead, the installation will succeed, but you will need to manually run `source .venv/bin/activate` afterwards.


### 2. Manual Installation

If you prefer to configure the environment manually:
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
To push generated data to the Hub or access gated models, set your access token:
```bash
export HF_TOKEN="ACCESS_TOKEN"
```
You can also pass directly the access token in `run.sh`



## Quick Start

### Basic Usage (Local Save)
This command reads a local file and saves the translation locally.

```bash
python src/inference.py \
    --model "soynade-research/Oolel-Small-v0.1" \
    --model_type "qwen2_5" \
    --input "data/sample_input.jsonl" \
    --split "train" \
    --text_column "input" \
    --system_prompt "Translate to Wolof the following sentence"
```

### Hub-to-Hub Usage
This command reads a dataset from HuggingFace and pushes the results back to the Hub.

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

-----
**Note**: For model type parameter, you can refer to [MS-SWIFT ](https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html) documentation for all supported models.

## Usage Examples

### 1. Local File → Local File

```bash
python inference.py \
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
    --input "user_name/repo" \
    --split "train" \
    --text_column "instruction" \
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


### 4. Using PyTorch Backend

```bash
python src/inference.py \
    --model "soynade-research/Oolel-Small-v0.1"\
    --backend pt \
    --model_type "qwen2_5" \
    --text_column "input" \
    --batch_size 4 \
    --input "data/input.jsonl" \
    --output "output/synthetic.jsonl"
```

## Advanced Usage

### Multi-GPU Inference (Tensor Parallelism)

To distribute the model across multiple GPUs, use the `--tensor_parallel_size` argument. This is required for vLLM to shard the model correctly.

```bash
python src/inference.py \
    --model "soynade-research/Oolel-Small-v0.1" \
    --backend vllm \
    --tensor_parallel_size 4 \
    --input "hf_user_name/dataset_name" \
    --split "train" \
    --output "output.jsonl"
```

### Memory Optimization

If you encounter Out of Memory (OOM) errors, adjust the GPU utilization or context window.

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

The pipeline accepts JSONL, JSON, or CSV. **JSONL (Recommended)**:

```jsonl
{"text": "Hello, how are you?", "id": 1}
{"text": "The weather is nice today.", "id": 2}
```

- JSON

```json
[
    {"text": "Hello, how are you?"},
    {"text": "The weather is nice today."}
]
```

- CSV

```csv
id,text
1,"Hello, how are you?"
2,"The weather is nice today."
```

## Output Format

Results are saved in JSONL format containing the prompt, original input, and model output.

```jsonl
{"system_prompt": "Translate to Wolof:", "input": "input", "output": "output"}
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
oolel-translator/
├── src/
│   └── inference.py    # Main script
├── README.md               
├── LICENSE.md               
├── CODE_OF_CONDUCT.md               
├── CONTRIBUTING.md              
├── pyproject.toml         # UV dependencies
├── setup.sh               # Setup script
├── .gitignore
├── data/
│   └── sample_input.jsonl
└── output/
    └── synthetic_wolof.jsonl # Example of output file
```


## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

Check [CONTRIBUTING.md](CONTRIBUTING.md) for more information. 

## License

MIT License - see LICENSE file for details

## Citation

```bibtex
@misc{oolel_translator,
  title={Oolel Translation Data Generation Pipeline},
  author={Soynade Research},
  year={2026},
  url={https://github.com/soynade-research/oolel-translator}
}
```
