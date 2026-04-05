# Synthetic data get

TODO:
- get all data sets and paths from: https://docs.google.com/presentation/d/1pI4JxBMnxcdXlWzzB6IkxqiCnbTD7Oubdg6eelA0DAE/edit#slide=id.g2f07874489c_5_0
    - and join them in an interleaved hf data set

- decide AF model
    - dsk-M
    - dsk-C
    - dsk-prover: https://huggingface.co/deepseek-ai/DeepSeek-Prover-V1.5-RL
    - let's test dsk-prover and if its the best and we joing all these data sets, can our alignment method select the best data?
- put model from lean4ai prompt to generate synthetic data
- generate the data and send to elyas

## Getting GAIR

Downloading the GAIR dataset do:
```bash
# create gair dir
mkdir ~/data/gair
# go to gair dir
~/data/gair
# get data
huggingface-cli download --resume-download --repo-type dataset GAIR/MathPile --local-dir /lfs/skampere1/0/brando9/data/gair --local-dir-use-symlinks False
```
Unzip the data
```bash
cd ~/data/gair
find . -type f -name "*.gz" -exec gzip -d {} \;

# Count lines
cd ~/data/gair
find . -type f -name "*.jsonl" | wc -l
# 41
find . -type f -name "*.jsonl" -exec cat {} + | wc -l
# 736,569
```
## Gen Synthetic Data
```bash
#TODO
python ...
```