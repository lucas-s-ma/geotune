#!/usr/bin/env python3
"""
Lite pipeline: CIF -> (LiteGearNet embeddings, Foldseek 3Di tokens) in AMPLIFY-style layout.

Outputs
-------
gearnet_embedding/
  protein0.npy, protein1.npy, ...
important_data/
  key_name2seq_token.npy           # { key: List[int] } AMPLIFY tokenizer ids
  key_name2foldseek_token.npy      # { key: List[int] } 3Di tokens
  key_names_valid_train.npy        # List[str]
  key_names_valid_valid.npy        # List[str]

Backend: Pure-PyTorch LiteGearNet (no TorchDrug extensions).
"""

import os, math, argparse, pathlib
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn

from transformers import AutoTokenizer
from Bio.PDB.MMCIFParser import MMCIFParser, MMCIF2Dict
from Bio.PDB.Polypeptide import is_aa, three_to_one
import mini3di

# ------------------------- config -------------------------
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA2IDX = {a: i for i, a in enumerate(AA_ORDER)}
UNK_IDX = len(AA_ORDER)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_TOKENIZER_ID = "chandar-lab/AMPLIFY_120M"

# common non-standard mappings (extend as needed)
NONSTD_MAP = {
    "MSE":"M", "SEC":"U", "PYL":"O", "HYP":"P",
    "SEP":"S", "TPO":"T", "PTR":"Y", "CSO":"C", "CSE":"C"
}

# 3-letter codes counted as amino acids in prescreen
AA3 = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
    "MSE","SEC","PYL","HYP","SEP","TPO","PTR","CSO","CSE"
}
# ----------------------------------------------------------

# -------------------- LiteGearNet model -------------------
class LiteGearNet(nn.Module):
    """Simple message-passing GNN (no TorchDrug C++)."""
    def __init__(self, dim=256, layers=3):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim)
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])

    def forward(self, x, edge_index):
        L, D = x.size()
        src, dst = edge_index
        h = self.in_proj(x)
        for mlp, norm in zip(self.mlps, self.norms):
            m = torch.zeros_like(h)
            m.index_add_(0, dst, h[src])  # aggregate neighbor features
            deg = torch.zeros(L, device=h.device).index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
            deg = deg.clamp_min_(1.0)
            m = m / deg.unsqueeze(-1)     # mean aggregation
            h = norm(h + mlp(m))          # residual
        return h
# ----------------------------------------------------------

# --------------------- parsing helpers --------------------
def _best_ca_atom(residue):
    """Return the CA atom, choosing highest-occupancy altloc if present."""
    if "CA" not in residue:
        return None
    ca = residue["CA"]
    try:
        # DisorderedAtom behaves like dict of altlocs
        best = max(ca.child_dict.values(), key=lambda a: (a.get_occupancy() or 0.0))
        return best
    except Exception:
        return ca

def parse_cif_concat_chains(cif_path: str, strict_poly_only: bool = True):
    """
    Parse an mmCIF file and return:
      - concatenated 1-letter amino-acid sequence
      - list of C-alpha coordinates (as numpy arrays)
      - Biopython Structure object

    Options
    -------
    strict_poly_only: keep chains whose entity.type is polypeptide(L) only.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", cif_path)
    cif_dict = MMCIF2Dict(cif_path)

    # Build map entity_id -> type, and asym_id (chain) -> entity_id
    ent_types = {}
    if "_entity.id" in cif_dict and "_entity.type" in cif_dict:
        ent_ids = cif_dict["_entity.id"]
        ent_types_list = cif_dict["_entity.type"]
        if not isinstance(ent_ids, list): ent_ids, ent_types_list = [ent_ids], [ent_types_list]
        ent_types = {eid: etype for eid, etype in zip(ent_ids, ent_types_list)}

    keep_asym = None
    if "_struct_asym.entity_id" in cif_dict:
        asym_ids = cif_dict["_struct_asym.id"]
        asym_ent_ids = cif_dict["_struct_asym.entity_id"]
        if not isinstance(asym_ids, list): asym_ids, asym_ent_ids = [asym_ids], [asym_ent_ids]
        keep_asym = set(
            a for a, eid in zip(asym_ids, asym_ent_ids)
            if (not strict_poly_only) or (ent_types.get(eid, "") == "polypeptide(L)")
        )

    sequence = []
    ca_coords = []
    kept_res, skipped_nonstd, skipped_noca = 0, 0, 0

    for model in structure:
        for chain in model:
            if keep_asym is not None and chain.id not in keep_asym:
                continue
            for residue in chain:
                # Only amino-acid residues (allow non-standards; filter later)
                if not is_aa(residue, standard=False):
                    continue

                resname = residue.get_resname().upper()

                # Map non-standards if possible; else try three_to_one
                if resname in NONSTD_MAP:
                    aa = NONSTD_MAP[resname]
                else:
                    try:
                        aa = three_to_one(resname)
                    except KeyError:
                        skipped_nonstd += 1
                        continue

                ca = _best_ca_atom(residue)
                if ca is None:
                    skipped_noca += 1
                    continue

                sequence.append(aa)
                ca_coords.append(ca.get_coord())
                kept_res += 1

    if not sequence:
        unique_comp_ids = set()
        if "_atom_site.label_comp_id" in cif_dict:
            comp_ids = cif_dict["_atom_site.label_comp_id"]
            if not isinstance(comp_ids, list): comp_ids = [comp_ids]
            unique_comp_ids = set(comp_ids)
        raise ValueError(
            f"No peptide residues parsed from {cif_path}. "
            f"Entity types: {sorted(set(ent_types.values()))}. "
            f"Unique comp_ids sample: {sorted(list(unique_comp_ids))[:20]}. "
            f"Check that this entry actually contains polypeptide(L) atoms."
        )

    return "".join(sequence), ca_coords, structure
# ----------------------------------------------------------

# ---------------------- graph builder ---------------------
def build_edges(L: int, ca_coords: List[np.ndarray], knn_k: int = 16):
    src, dst = [], []
    # sequential edges
    for i in range(L - 1):
        src += [i, i+1]; dst += [i+1, i]
    # kNN edges
    valid = [i for i,c in enumerate(ca_coords) if not np.isnan(c).any()]
    if len(valid) >= 2:
        coords = np.stack([ca_coords[i] for i in valid], 0)
        d2 = np.sum((coords[None]-coords[:,None])**2, -1)
        for a_pos, i in enumerate(valid):
            nbrs = [valid[j] for j in np.argsort(d2[a_pos])[1:knn_k+1]]
            for j in nbrs:
                src += [i, j]; dst += [j, i]
    return torch.tensor([src, dst], dtype=torch.long)
# ----------------------------------------------------------

# ------------------- embedding function -------------------
@torch.no_grad()
def encode_lite_gearnet(seq: str, ca_coords: List[np.ndarray], hidden_dim=256, knn_k=16):
    idx = torch.tensor([AA2IDX.get(a, UNK_IDX) for a in seq], dtype=torch.long, device=DEVICE)
    emb = nn.Embedding(UNK_IDX+1, hidden_dim).to(DEVICE).eval()
    x = emb(idx)                          # [L, D]
    edge_index = build_edges(len(seq), ca_coords, knn_k).to(DEVICE)
    model = LiteGearNet(dim=hidden_dim).to(DEVICE).eval()
    return model(x, edge_index).cpu().numpy()
# ----------------------------------------------------------

# ---------------------- 3Di tokens ------------------------
def encode_3di_tokens(structure) -> List[int]:
    encoder = mini3di.Encoder()
    tokens = []
    model = next(structure.get_models())
    for chain in model:
        try:
            t = encoder.encode_chain(chain)
            if t is not None and len(t) > 0:
                tokens.extend(t.astype(int).tolist())
        except Exception:
            continue
    return tokens
# ----------------------------------------------------------

# ---------------------- utilities -------------------------
def ensure_same_length(a: List[int], b_len: int) -> List[int]:
    if len(a) == b_len: return a
    if len(a) > b_len: return a[:b_len]
    if len(a) == 0: return [0] * b_len
    return a + [a[-1]] * (b_len - len(a))
# ----------------------------------------------------------

# --------------------- prescreen helper -------------------
def has_protein_residues(cif_path: str, min_len: int = 1) -> bool:
    """
    Fast mmCIF check: returns True iff there are >= min_len peptide residues.
    Avoids full Structure build; inspects comp_ids in _atom_site.
    """
    try:
        d = MMCIF2Dict(cif_path)
    except Exception:
        return False

    comp_ids = d.get("_atom_site.label_comp_id")
    if comp_ids is None:
        return False
    if not isinstance(comp_ids, list):
        comp_ids = [comp_ids]

    aa_count = sum(1 for c in comp_ids if c.upper() in AA3)
    return aa_count >= min_len
# ----------------------------------------------------------

# ---------------------------- main ------------------------
def main():
    p = argparse.ArgumentParser(description="LiteGearNet embeddings + 3Di tokens (AMPLIFY layout)")
    p.add_argument("--cif_dir", type=str, default="cifs")
    p.add_argument("--out_emb_dir", type=str, default="co-amp/dataprep/gearnet_embedding")
    p.add_argument("--out_meta_dir", type=str, default="co-amp/dataprep/important_data")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--knn_k", type=int, default=16)
    p.add_argument("--valid_fraction", type=float, default=0.0)
    p.add_argument("--min_len", type=int, default=10, help="Min AA residues required to keep a CIF")
    p.add_argument("--strict_poly_only", action="store_true", help="Keep only chains with entity.type polypeptide(L)")
    args = p.parse_args()

    os.makedirs(args.out_emb_dir, exist_ok=True)
    os.makedirs(args.out_meta_dir, exist_ok=True)

    cif_paths = sorted([str(pth) for pth in pathlib.Path(args.cif_dir).glob("*.cif")])
    if not cif_paths:
        raise SystemExit(f"No .cif files found under {args.cif_dir}")

    # ---------- prescreen ----------
    screened = []
    for pth in cif_paths:
        if has_protein_residues(pth, min_len=args.min_len):
            screened.append(pth)
        else:
            print(f"[skip:non-protein] {pth}")
    if not screened:
        raise SystemExit("No protein-like CIFs after prescreening. Try lowering --min_len.")

    key2seq_tokens, key2foldseek_tokens, all_keys = {}, {}, []
    kept = skipped = 0

    for i, cif_path in enumerate(screened):
        key = f"protein{kept}"   # index only successful ones
        try:
            seq, ca_coords, structure = parse_cif_concat_chains(
                cif_path, strict_poly_only=args.strict_poly_only
            )

            node_repr = encode_lite_gearnet(
                seq, ca_coords,
                hidden_dim=args.hidden_dim, knn_k=args.knn_k
            )

            foldseek_tok = encode_3di_tokens(structure)

            # align lengths
            L = node_repr.shape[0]
            foldseek_tok = ensure_same_length(foldseek_tok, L)

            np.save(os.path.join(args.out_emb_dir, f"{key}.npy"), node_repr)
            key2seq_tokens[key] = seq
            clean_foldseek_tok = [0 if x is None else x for x in foldseek_tok]
            key2foldseek_tokens[key] = np.array(clean_foldseek_tok, dtype=np.int64)
            all_keys.append(key)

            kept += 1
            print(f"[{kept}/{len(screened)} ok] {key} from {os.path.basename(cif_path)}: L={L}")
        except Exception as e:
            skipped += 1
            print(f"[skip:error] {os.path.basename(cif_path)} → {e}")

    # ---------- splits & metadata ----------
    n_valid = int(round(len(all_keys) * args.valid_fraction))
    valid_keys, train_keys = all_keys[:n_valid], all_keys[n_valid:]
    np.save(os.path.join(args.out_meta_dir, "key_names_valid_train.npy"), np.array(train_keys, dtype=object))
    np.save(os.path.join(args.out_meta_dir, "key_names_valid_valid.npy"), np.array(valid_keys, dtype=object))
    np.save(os.path.join(args.out_meta_dir, "key_name2seq_token.npy"), key2seq_tokens, allow_pickle=True)
    np.save(os.path.join(args.out_meta_dir, "key_name2foldseek_token.npy"), key2foldseek_tokens, allow_pickle=True)

    print(f"\nSummary: kept={kept}, skipped={skipped}, total_input={len(cif_paths)}")
    print("Done. Wrote embeddings + maps into AMPLIFY-style layout.")

if __name__ == "__main__":
    main()
