import os
import requests

pdb_ids_file = "pdb_ids.txt"
out_dir = "dataprep/cifs"
fasta_dir = "fastas"

os.makedirs(out_dir, exist_ok=True)
os.makedirs(fasta_dir, exist_ok=True)

# Step 1: read IDs
with open(pdb_ids_file) as f:
    pdb_ids = [
        line.strip().split(",")[0]   # handles potential CSV-like entries
        for line in f.readlines()[2:]  # skip headers
        if line.strip() and not line.startswith("IDCODE")
    ]

print(f"Found {len(pdb_ids)} PDB IDs")

# Step 2: download CIF + FASTA
for pdb_id in pdb_ids:
    pdb_id = pdb_id.upper()  # ensure uppercase
    try:
        # Check if the entry is a protein-only structure
        info_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        info_r = requests.get(info_url)
        if info_r.status_code == 200:
            info_data = info_r.json()
            entry_info = info_data.get("rcsb_entry_info", {})
            is_protein = entry_info.get("polymer_entity_count_protein", 0) > 0
            has_dna = entry_info.get("polymer_entity_count_dna", 0) > 0
            has_rna = entry_info.get("polymer_entity_count_rna", 0) > 0

            if not is_protein or has_dna or has_rna:
                print(f"Skipping {pdb_id}: Not a protein-only structure.")
                continue
        else:
            print(f"Could not verify molecule type for {pdb_id}, skipping.")
            continue

        # Download CIF
        cif_url = f"https://files.rcsb.org/download/{pdb_id}.cif"


        # Download FASTA
        fasta_url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
        fasta_out = os.path.join(fasta_dir, f"{pdb_id}.fasta")
        r = requests.get(fasta_url)
        if r.status_code == 200:
            with open(fasta_out, "wb") as f:
                f.write(r.content)
            print(f"Downloaded FASTA for {pdb_id}")
        else:
            print(f"Failed FASTA for {pdb_id} (status {r.status_code})")

    except Exception as e:
        print(f"Error downloading {pdb_id}: {e}")
