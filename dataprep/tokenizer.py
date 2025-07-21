import numpy as np
from transformers import AutoTokenizer
import os

# Ensure the directory exists
os.makedirs("important_data", exist_ok=True)

# Define 10 example protein sequences
sequences = [
    "MSTAAALYIYSANQ",
    "MVLSPADKTNVKAAW",
    "MNIKDTLLDGVVAEY",
    "MITLRVGVPTLKRSF",
    "MDKDLKVDLKKAAL",
    "MFRHRIYGTLAPVSR",
    "MNKATIVGDPTLAAN",
    "MVYAKRKAAKPGNYL",
    "MKVKLFFWMRNVKAI",
    "MNQPLVPAANHGSKE"
]

# Initialize the tokenizer from AMPLIFY's pretrained model
tokenizer = AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_120M", trust_remote_code=True)

# Tokenize each sequence
seq_dict = {f"protein{i}": tokenizer(seq)["input_ids"] for i, seq in enumerate(sequences)}

# 1) Build key list
keys = [f"protein{i}" for i in range(len(sequences))]
np.save("important_data/key_names_train.npy", np.array(keys))

# Generate dummy embeddings (512-dimensional random vectors)
dummy_embeddings = {key: np.random.rand(512) for key in seq_dict.keys()}

# Save the dummy embeddings
np.save("important_data/key_name2foldseek_token.npy", dummy_embeddings)

# Generate dummy labels (e.g., binary labels for fine-tuning)
np.save("important_data/seq_labels.npy", np.random.randint(0, 2, size=(len(sequences),)))
np.save("important_data/struc_labels.npy", np.random.randint(0, 2, size=(len(sequences),)))

train_keys = np.load('important_data/key_names_train.npy')

# Split the dataset for validation (e.g., 10% of the data for validation)
validation_size = int(len(train_keys) * 0.1)  # 10% for validation
validation_keys = train_keys[:validation_size]  # Select the first 10% (or randomly shuffle if needed)


# 2) For each sequence, tokenize to STRING tokens
#    (not IDs)
key_name2seq_token = {}
for key, seq in zip(keys, sequences):
    # You can either split by character:
    #    token_list = list(seq)
    # or use the HF tokenizer’s own split:
    token_list = tokenizer.tokenize(seq, add_special_tokens=False)
    key_name2seq_token[key] = token_list

# Save as a pickleable object array
np.save(
    "important_data/key_name2seq_token.npy",
    key_name2seq_token,
    allow_pickle=True
)

# 3) (Optional) Dummy foldseek tokens & embeddings stay as before
#    You already have key_name2foldseek_token.npy, seq_labels.npy, etc.

# 4) Validation splits
train_keys = np.array(keys)
np.save("important_data/key_names_valid_train.npy", train_keys[:1])   # e.g., 1 val
np.save("important_data/key_names_valid_valid.npy", train_keys[:1])