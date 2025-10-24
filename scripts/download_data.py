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


def download_protein_data(pdb_ids, output_dir, include_chains=None):
    """
    Download protein structure data from RCSB PDB
    
    Args:
        pdb_ids: List of PDB IDs to download
        output_dir: Output directory to save files2
        include_chains: List of chain IDs to include (None for all chains)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    pdbl = PDBList()
    
    success_count = 0
    fail_count = 0
    
    for pdb_id in pdb_ids:
        pdb_id = pdb_id.strip().upper()
        print(f"Downloading {pdb_id}...")

        # check if this is already in output_dir
        if os.path.exists(os.path.join(output_dir, f"{pdb_id}.pdb")):
            print(f"{pdb_id} already exists in {output_dir}, skipping download.")
            success_count += 1
            continue
        
        # check if this is already in output_dir
        if os.path.exists(os.path.join(output_dir, f"{pdb_id}.pdb")):
            print(f"{pdb_id} already exists in {output_dir}, skipping download.")
            success_count += 1
            continue
        


        try:
            # Check if the entry is a protein-only structure
            info_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
            info_r = requests.get(info_url)
            if info_r.status_code == 200:
                info_data = info_r.json()
                entry_info = info_data.get("rcsb_entry_info", {})
                
                # Check if it's a protein structure
                is_protein = entry_info.get("polymer_entity_count_protein", 0) > 0
                has_dna = entry_info.get("polymer_entity_count_dna", 0) > 0
                has_rna = entry_info.get("polymer_entity_count_rna", 0) > 0
                
                if not is_protein or has_dna or has_rna:
                    print(f"Skipping {pdb_id}: Not a protein-only structure.")
                    continue
            else:
                print(f"Could not verify molecule type for {pdb_id}, skipping.")
                continue
            
            # Download the PDB file
            pdb_filename = pdbl.retrieve_pdb_file(
                pdb_id, 
                pdir=output_dir, 
                file_format='pdb',
                overwrite=True
            )
            
            # Rename to consistent format
            old_path = os.path.join(output_dir, f"pdb{pdb_id.lower()}.ent")
            new_path = os.path.join(output_dir, f"{pdb_id}.pdb")
            
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
                print(f"Downloaded {pdb_id} to {new_path}")
                success_count += 1
            else:
                print(f"Failed to download {pdb_id}")
                fail_count += 1
                
        except Exception as e:
            print(f"Error downloading {pdb_id}: {e}")
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


def main():
    parser = argparse.ArgumentParser(description="Download protein structure data")
    parser.add_argument("--output_dir", type=str, required=True, 
                       help="Output directory to save PDB files")
    parser.add_argument("--pdb_list", type=str, 
                       help="Path to file containing PDB IDs (one per line)")
    parser.add_argument("--uniprot_list", type=str,
                       help="Path to file containing UniProt IDs (one per line)")
    parser.add_argument("--example", action="store_true",
                       help="Download example PDB structures")
    
    args = parser.parse_args()
    
    if args.example:
        # Example PDB IDs that are known to be good protein structures
        example_pdbs = [
            "1TIM",  # Triosephosphate isomerase
            "2V7V",  # Small protein
            "3ZJE",  # Another small protein
            "4GCR",  # Cytochrome C
            "1BPI",  # Barnase
        ]
        print("Downloading example PDB structures...")
        download_protein_data(example_pdbs, args.output_dir)
        
    elif args.pdb_list:
        with open(args.pdb_list, 'r') as f:
            pdb_ids = [line.strip() for line in f.readlines() if line.strip()]
        download_protein_data(pdb_ids, args.output_dir)
        
    elif args.uniprot_list:
        with open(args.uniprot_list, 'r') as f:
            uniprot_ids = [line.strip() for line in f.readlines() if line.strip()]
        download_by_uniprot(uniprot_ids, args.output_dir)
        
    else:
        print("Please specify either --pdb_list, --uniprot_list, or --example")


if __name__ == "__main__":
    main()