#!/bin/bash

# run `accelerate config` first. pass --deepspeed to finetune.py if using DeepSpeed

# python3 truncate.py 8192 output/truncated-8k

accelerate launch finetune.py \
    --output-dir output/yarn-7b-64k \
    --model cognitivecomputations/TinyDolphin-2.8-1.1b \
    --max-train-steps 30 \
    --dataset emozilla/pg_books-tokenized-bos-eos-chunked-65536