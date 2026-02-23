"""
Download protein structure data for training
"""
import os
import sys
import requests
import argparse
from Bio.PDB import PDBList
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from tqdm import tqdm

def download_protein_data(pdb_ids, output_dir, include_chains=None):
    """
    Download protein structure data from RCSB PDB.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdbl = PDBList()
    
    success_count = 0
    fail_count = 0
    
    for pdb_id in tqdm(pdb_ids, desc="Downloading PDB files"):
        pdb_id = pdb_id.strip().upper()

        # Check for both naming conventions
        new_path = os.path.join(output_dir, f"{pdb_id}.pdb")
        print(pdb_id)
        old_path_check = os.path.join(output_dir, f"pdb{pdb_id.lower()}.ent")
        
        if os.path.exists(new_path):
            success_count += 1
            continue
        
        if os.path.exists(old_path_check):
            # File was downloaded but not renamed - rename it now
            os.rename(old_path_check, new_path)
            success_count += 1
            continue

        try:
            info_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
            info_r = requests.get(info_url)
            if info_r.status_code == 200:
                info_data = info_r.json()
                entry_info = info_data.get("rcsb_entry_info", {})
                if not entry_info.get("polymer_entity_count_protein", 0) > 0 or \
                   entry_info.get("polymer_entity_count_dna", 0) > 0 or \
                   entry_info.get("polymer_entity_count_rna", 0) > 0:
                    continue
            else:
                continue

            pdbl.retrieve_pdb_file(pdb_id, pdir=output_dir, file_format='pdb', overwrite=False)
            
            old_path = os.path.join(output_dir, f"pdb{pdb_id.lower()}.ent")
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
                success_count += 1
            else:
                fail_count += 1
                
        except Exception as e:
            fail_count += 1
    
    print(f"\nDownload completed! Success: {success_count}, Failed: {fail_count}")


def download_by_uniprot(uniprot_ids, output_dir):
    """
    Download PDB structures by UniProt ID
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for uniprot_id in uniprot_ids:
        print(f"Finding PDB structures for UniProt ID: {uniprot_id}")
        
        # Query RCSB to get associated PDB IDs
        search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
        
        query_json = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                    "operator": "exact_match",
                    "value": uniprot_id,
                    "negation": False
                }
            },
            "request_options": {
                "return_all_hits": True
            },
            "return_type": "polymer_entity"
        }
        
        try:
            response = requests.post(search_url, json=query_json)
            if response.status_code == 200:
                results = response.json()
                pdb_ids = []
                
                if "result_set" in results:
                    for result in results["result_set"]:
                        pdb_id = result["identifier"].split(".")[0]  # Extract PDB ID
                        if pdb_id not in pdb_ids:
                            pdb_ids.append(pdb_id)
                
                print(f"Found {len(pdb_ids)} PDB structures for {uniprot_id}")
                
                # Download the found PDB files
                download_protein_data(pdb_ids[:10], output_dir)  # Limit to first 10 structures
                
        except Exception as e:
            print(f"Error searching for UniProt {uniprot_id}: {e}")


def validate_downloaded_files(output_dir):
    """
    Validates downloaded PDB files.
    """
    print(f"Validating files in {output_dir}...")
    valid_count = 0
    invalid_count = 0
    parser = PDBParser(QUIET=True)
    
    for filename in os.listdir(output_dir):
        if filename.endswith(".pdb"):
            try:
                parser.get_structure("protein", os.path.join(output_dir, filename))
                valid_count += 1
            except Exception as e:
                invalid_count += 1
    
    print(f"Validation complete: {valid_count} valid, {invalid_count} invalid PDB files.")


def main():
    parser = argparse.ArgumentParser(
        description="Download and validate protein structure data from RCSB PDB."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory to save PDB files."
    )
    parser.add_argument(
        "--pdb_list", 
        type=str, 
        help="Path to a file containing a list of PDB IDs to download."
    )
    parser.add_argument(
        "--uniprot_list", 
        type=str,
        help="Path to a file containing a list of UniProt IDs to download."
    )
    parser.add_argument(
        "--example", 
        action="store_true",
        help="Download a small set of example PDB structures."
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the downloaded PDB files."
    )
    
    args = parser.parse_args()
    
    if args.example:
        example_pdbs = ["1TIM", "2V7V", "3ZJE", "4GCR", "1BPI"]
        download_protein_data(example_pdbs, args.output_dir)
        
    elif args.pdb_list:
        try:
            with open(args.pdb_list, 'r') as f:
                pdb_ids = [line.strip() for line in f if line.strip()]
            download_protein_data(pdb_ids, args.output_dir)
        except FileNotFoundError:
            print(f"Error: PDB list file not found at {args.pdb_list}")
            exit(1)
        
    elif args.uniprot_list:
        try:
            with open(args.uniprot_list, 'r') as f:
                uniprot_ids = [line.strip() for line in f if line.strip()]
            download_by_uniprot(uniprot_ids, args.output_dir)
        except FileNotFoundError:
            print(f"Error: UniProt list file not found at {args.uniprot_list}")
            exit(1)
        
    else:
        print("Please specify a source for PDB IDs (--pdb_list, --uniprot_list, or --example).")

    if args.validate:
        validate_downloaded_files(args.output_dir)


if __name__ == "__main__":
    main()