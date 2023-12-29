from urllib.request import urlretrieve
import pandas as pd
import os
from tqdm import tqdm

# Download the parquet table
if not os.path.exists("metadata.parquet"):
    table_url = f"https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet"
    urlretrieve(table_url, "metadata.parquet")

# Read the table using Pandas
prompts = pd.read_parquet("metadata.parquet", columns=["prompt"])
prompts = prompts[: len(prompts) // 2]
print(prompts.shape)
# print(prompts.head())
print(type(prompts))

unique_prompts = set()
for prompt in tqdm(prompts.itertuples()):
    unique_prompts.add(prompt[1])

print("unique prompts", len(unique_prompts))
with open("diffusion_db_prompts.txt", "w") as f:
    for p in unique_prompts:
        f.write(p + "\n")
