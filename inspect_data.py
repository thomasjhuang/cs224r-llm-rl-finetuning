from datasets import load_from_disk
import numpy as np

ds = load_from_disk("data/smoltalk_tokenized")
lengths = [len(x["input_ids"]) for x in ds]
print("Average length:", np.mean(lengths))
print("90th percentile:", np.percentile(lengths, 90))
print("Max length observed:", max(lengths))

