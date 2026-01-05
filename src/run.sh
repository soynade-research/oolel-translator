python src/inference.py \
    --model "soynade-research/Oolel-Small-v0.1" \
    --input "data/sample_input.jsonl" \
    --split "train" \
    --text_column "input" \
    --output "output/synthetic_wolof.jsonl" \
    --system_prompt "Translate the following English text into standard, grammatically correct Wolof. Use official orthography."