#!/bin/sh
git credential approve << EOT 
url=https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
username=alperd
password=$HF_TOKEN
EOT