# File: ai/eqbench3/utils/merge_candidates.py

import os
import sys
import logging
import json
import gzip
import re
import copy
from datetime import datetime, timezone
import argparse
from collections import defaultdict
from core.elo import (
    get_solver_comparisons,
    models_in_comparisons,
)
from typing import Dict, Any

# --- Logging Setup ---
def setup_merge_logging(level_str):
    log_level = getattr(logging, level_str.upper(), logging.INFO)

    # Re-initialise logging even if it was configured earlier
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True              # <-- key line
    )

    # If other modules grabbed the root logger before this, make sure they honour the new level.
    logging.getLogger().setLevel(log_level)

    logging.debug(f"Logging level set to {level_str.upper()}")
# --- End Logging Setup ---



# Add the parent directory (ai/eqbench3) to sys.path to allow imports like 'from utils. ...'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    # Now these imports should work when run from ai/eqbench3/
    from utils.file_io import load_json_file, save_json_file
    import utils.constants as C
    from core.trueskill_solver import solve_with_trueskill, normalize_elo_scores
    from core.elo_helpers import should_ignore_scenario
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Original sys.path:", sys.path)
    print("Please ensure the script is run from the 'ai/eqbench3/' directory.")
    sys.exit(1)
# --- End Imports ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_merge_candidates(local_runs, local_elo, canonical_runs, canonical_elo):
    """Identifies runs in local files that meet merge criteria."""
    candidates = []
    processed_model_names = set() # Track models already added as candidates

    # Build set of model names already in canonical data (runs or elo)
    canonical_model_names = set(k for k in canonical_elo if k != "__metadata__")
    for run_data in canonical_runs.values():
        if isinstance(run_data, dict):
            canonical_model_names.add(run_data.get("model_name", run_data.get("test_model")))

    logging.info(f"Found {len(canonical_model_names)} unique model names in canonical data.")
    logging.debug(f"Total items found in local_runs: {len(local_runs)}") # <-- ADDED DEBUG

    for run_key, run_data in local_runs.items():
        logging.debug(f"Processing local run key: '{run_key}'") # <-- ADDED DEBUG

        if not isinstance(run_data, dict):
            logging.debug(f"Skipping '{run_key}': Run data is not a dictionary.") # <-- ADDED DEBUG
            continue

        model_name = run_data.get("model_name", run_data.get("test_model"))
        if not model_name:
            logging.warning(f"Skipping run {run_key}: Missing model name.") # Keep as warning
            continue

        # Avoid adding the same model multiple times if it has multiple local runs
        if model_name in processed_model_names:
            logging.debug(f"Skipping '{run_key}' (model '{model_name}'): Model name already processed from a previous run key.") # <-- ADDED DEBUG
            continue

        # --- Check Criteria ---
        # 1. Exists in local ELO?
        if model_name not in local_elo or not isinstance(local_elo.get(model_name), dict):
            logging.debug(f"Skipping {model_name} ({run_key}): Missing or invalid entry in local ELO file.")
            continue

        # 2. Completeness (Rubric, ELO scores, Matchups)
        results = run_data.get("results", {})
        rubric_score = results.get("average_rubric_score")
        elo_raw = results.get("elo_raw")
        has_rubric = rubric_score is not None and rubric_score != "Skipped" and results.get("rubric_error") is None
        has_elo = elo_raw is not None and elo_raw != "Skipped" and results.get("elo_error") is None

        if not has_rubric:
            logging.debug(f"Skipping {model_name} ({run_key}): Missing valid Rubric score.")
            continue
        if not has_elo:
            logging.debug(f"Skipping {model_name} ({run_key}): Missing valid ELO score.")
            continue

        # Check for matchups involving this model in local ELO comparisons
        has_matchups = False
        local_comps = local_elo.get("__metadata__", {}).get("global_pairwise_comparisons", [])
        for comp in local_comps:
            pair = comp.get("pair", {})
            if model_name in (pair.get("test_model"), pair.get("neighbor_model")):
                has_matchups = True
                break
        if not has_matchups:
            logging.debug(f"Skipping {model_name} ({run_key}): No matchups found in local ELO comparisons.")
            continue

        # 3. No Name Collision in Canonical Data
        if model_name in canonical_model_names:
            logging.debug(f"Skipping {model_name} ({run_key}): Name already exists in canonical data.")
            continue

        # --- Candidate Found ---
        candidates.append({
            "run_key": run_key,
            "model_name": model_name,
            "rubric_score": rubric_score * 5.0 if isinstance(rubric_score, (int, float)) else "N/A",
            "elo_norm": results.get("elo_normalized", "N/A")
        })
        processed_model_names.add(model_name)
        logging.info(f"Found potential candidate: {model_name} (from run {run_key})")

    return candidates

def select_candidates(candidates):
    """Prompts the user to select candidates from a list."""
    if not candidates:
        return []

    print("\n--- Merge Candidates ---")
    for i, cand in enumerate(candidates):
        print(f"{i+1: >3}. {cand['model_name']} (Run: {cand['run_key']}, Rubric: {cand['rubric_score']:.1f}, ELO Norm: {cand['elo_norm']})")
    print("----------------------")

    while True:
        try:
            selection = input("Enter numbers of candidates to merge (e.g., 1,3,4), 'all', or 'none': ").strip().lower()
            if selection == 'none':
                return []
            if selection == 'all':
                return candidates # Return all candidate dicts

            selected_indices = set()
            parts = selection.split(',')
            for part in parts:
                part = part.strip()
                if not part: continue
                index = int(part) - 1
                if 0 <= index < len(candidates):
                    selected_indices.add(index)
                else:
                    print(f"Invalid number: {part}. Please enter numbers between 1 and {len(candidates)}.")
                    raise ValueError("Invalid index")

            # Return the selected candidate dicts
            return [candidates[i] for i in sorted(list(selected_indices))]

        except ValueError:
            print("Invalid input. Please use the specified format.")
        except Exception as e:
            print(f"An error occurred: {e}")

def merge_data(selected_candidates, local_runs, local_elo, canonical_runs, canonical_elo):
    """Moves selected run data and relevant comparisons to canonical files."""
    if not selected_candidates:
        return False # Indicate nothing was merged

    merged_model_names = set(c['model_name'] for c in selected_candidates)
    logging.info(f"Preparing to merge {len(merged_model_names)} models: {', '.join(merged_model_names)}")

    # Determine the final set of models that will be in the canonical ELO file
    final_canonical_models = set(k for k in canonical_elo if k != "__metadata__")
    final_canonical_models.update(merged_model_names)
    logging.info(f"Final canonical ELO file will contain {len(final_canonical_models)} models.")

    # --- Process Local ELO Comparisons ---
    local_comps = local_elo.get("__metadata__", {}).get("global_pairwise_comparisons", [])
    comps_to_move = []
    comps_to_keep_local = []
    moved_comp_count = 0

    for comp in local_comps:
        pair = comp.get("pair", {})
        model_a = pair.get("test_model")
        model_b = pair.get("neighbor_model")

        # Check if this comparison involves one of the models being merged
        comp_involves_merged_model = model_a in merged_model_names or model_b in merged_model_names

        if comp_involves_merged_model:
            # Check if BOTH models in the pair will be in the final canonical set
            if model_a in final_canonical_models and model_b in final_canonical_models:
                comps_to_move.append(comp)
                moved_comp_count += 1
            else:
                # Keep comparison locally if it involves a merged model but the other model isn't canonical
                comps_to_keep_local.append(comp)
                logging.debug(f"Keeping comparison locally (one model not canonical): {model_a} vs {model_b}")
        else:
            # Keep comparison locally if it doesn't involve any model being merged now
            comps_to_keep_local.append(comp)

    logging.info(f"Identified {moved_comp_count} comparisons to move to canonical ELO.")
    logging.info(f"{len(comps_to_keep_local)} comparisons will remain in local ELO.")

    # Update local ELO comparisons
    if "__metadata__" not in local_elo: local_elo["__metadata__"] = {}
    local_elo["__metadata__"]["global_pairwise_comparisons"] = comps_to_keep_local

    # Add comparisons to canonical ELO
    if "__metadata__" not in canonical_elo: canonical_elo["__metadata__"] = {}
    if "global_pairwise_comparisons" not in canonical_elo["__metadata__"]:
        canonical_elo["__metadata__"]["global_pairwise_comparisons"] = []
    canonical_elo["__metadata__"]["global_pairwise_comparisons"].extend(comps_to_move)

    # --- Move Run Data and ELO Entries ---
    for candidate in selected_candidates:
        run_key = candidate["run_key"]
        model_name = candidate["model_name"]

        # Move run data
        if run_key in local_runs:
            canonical_runs[run_key] = local_runs[run_key]
            del local_runs[run_key]
            logging.info(f"Moved run data for {run_key} to canonical runs.")
        else:
            logging.warning(f"Run key {run_key} not found in local runs data during merge.")

        # Move ELO entry
        if model_name in local_elo:
            if model_name != "__metadata__": # Safety check
                canonical_elo[model_name] = local_elo[model_name]
                del local_elo[model_name]
                logging.info(f"Moved ELO entry for {model_name} to canonical ELO.")
        else:
            # This might happen if the ELO entry was already moved by another run of the same model
            logging.warning(f"Model name {model_name} not found in local ELO data during merge (might be ok if merged previously).")

    return True # Indicate merging occurred

# --- ELO Re-calculation -------------------------------------------------
def recalculate_canonical_elo(canonical_elo) -> bool:
    """
    Recalculate canonical ELO ratings.
    Uses the same comparison-filter pipeline as run_elo_analysis_eqbench3.
    """
    logging.info("Recalculating ELO ratings for the updated canonical dataset…")

    DEFAULT_ELO      = getattr(C, "DEFAULT_ELO", 1500)
    TS_DEFAULT_SIGMA = 350 / 3
    RANK_WINDOW      = 8

    try:
        meta            = canonical_elo.get("__metadata__", {})
        all_comparisons = meta.get("global_pairwise_comparisons", [])
        all_models      = [m for m in canonical_elo if m != "__metadata__"]

        if not all_models:
            logging.warning("No models in canonical ELO.")
            return False

        if not all_comparisons:
            logging.warning("No comparisons; assigning defaults.")
            for m in all_models:
                canonical_elo[m] = {
                    "elo":      DEFAULT_ELO,
                    "elo_norm": DEFAULT_ELO,
                    "sigma":    TS_DEFAULT_SIGMA,
                    "ci_low":   DEFAULT_ELO - 1.96 * TS_DEFAULT_SIGMA,
                    "ci_high":  DEFAULT_ELO + 1.96 * TS_DEFAULT_SIGMA,
                }
            return True

        # ---------- identical pipeline to main ELO run ------------------
        initial_snapshot = {
            m: canonical_elo.get(m, {}).get("elo", DEFAULT_ELO) for m in all_models
        }
        comps_for_solver = get_solver_comparisons(
            all_comparisons,
            initial_snapshot,
            rank_window=RANK_WINDOW,
        )
        if not comps_for_solver:
            logging.warning("No comparisons after rank-window filter.")
            return False

        models_for_solver = sorted(set(all_models) | models_in_comparisons(comps_for_solver))

        mu_map, sigma_map = solve_with_trueskill(
            models_for_solver,
            comps_for_solver,
            {m: DEFAULT_ELO for m in models_for_solver},
            debug=False,
            use_fixed_initial_ratings=True,
            bin_size=1,
            return_sigma=True,
        )
        mu_norm_map = normalize_elo_scores(mu_map)

        # ---------- write results back ----------------------------------
        for m in models_for_solver:
            mu_raw  = mu_map.get(m, DEFAULT_ELO)
            sigma   = sigma_map.get(m, TS_DEFAULT_SIGMA)
            ci_low  = mu_raw - 1.96 * sigma
            ci_high = mu_raw + 1.96 * sigma
            canonical_elo[m] = {
                "elo":       round(mu_raw, 2),
                "elo_norm":  round(mu_norm_map.get(m, DEFAULT_ELO), 2),
                "sigma":     round(sigma, 2),
                "ci_low":    round(ci_low, 2),
                "ci_high":   round(ci_high, 2),
            }

        # ---------- normalise CI bounds ---------------------------------
        raw_plus_bounds = {
            **{m: d["elo"] for m, d in canonical_elo.items() if m != "__metadata__"},
            **{f"{m}__low":  d["ci_low"]  for m, d in canonical_elo.items() if m != "__metadata__"},
            **{f"{m}__high": d["ci_high"] for m, d in canonical_elo.items() if m != "__metadata__"},
        }
        norm_bounds = normalize_elo_scores(raw_plus_bounds)
        for m, d in canonical_elo.items():
            if m != "__metadata__":
                d["ci_low_norm"]  = round(norm_bounds.get(f"{m}__low",  d["elo_norm"]), 2)
                d["ci_high_norm"] = round(norm_bounds.get(f"{m}__high", d["elo_norm"]), 2)

        logging.info("Canonical ELO recalculation complete.")
        return True

    except Exception:
        logging.error("ELO recalculation failed", exc_info=True)
        return False

import uuid, shutil

def _atomic_multi_save(path_to_data: Dict[str, Any]) -> bool:
    """
    Transactionally write several JSON blobs.  If anything fails,
    originals are untouched and all temp files are removed.
    """
    temps = {}

    # ---- 1. write each temp file ---------------------------------------
    for final_path, data in path_to_data.items():
        dir_name = os.path.dirname(final_path)
        stem, ext = os.path.splitext(os.path.basename(final_path))
        if ext == ".gz":           # keep .json.gz intact
            stem2, ext2 = os.path.splitext(stem)
            tmp_name = f"{stem2}.tmp.{uuid.uuid4().hex}{ext2}.gz"
        else:
            tmp_name = f"{stem}.tmp.{uuid.uuid4().hex}{ext}"
        tmp_path = os.path.join(dir_name, tmp_name)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        if not save_json_file(data, tmp_path):
            # ---- cleanup temps already written ----
            for t in temps.values():
                try: os.remove(t)
                except OSError: pass
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except OSError: pass
            return False

        temps[final_path] = tmp_path

    # ---- 2. rename temps onto finals -----------------------------------
    try:
        for final_path, tmp_path in temps.items():
            os.replace(tmp_path, final_path)
        return True

    except Exception as e:
        logging.error(f"Multi-save rename phase failed: {e}", exc_info=True)
        # ---- cleanup all temps (none renamed yet if replace is atomic) ---
        for t in temps.values():
            try: os.remove(t)
            except OSError: pass
        return False



# -------------------------------------------------------------------------



def main():
    parser = argparse.ArgumentParser(
        description="Identify and merge candidate runs from local files to canonical leaderboard files."
    )
    parser.add_argument(
        "--local-runs",
        default=C.DEFAULT_LOCAL_RUNS_FILE,
        help="Path to the local runs JSON file.",
    )
    parser.add_argument(
        "--local-elo",
        default=C.DEFAULT_LOCAL_ELO_FILE,
        help="Path to the local ELO JSON file.",
    )
    parser.add_argument(
        "--canonical-runs",
        default=C.CANONICAL_LEADERBOARD_RUNS_FILE,
        help="Path to the canonical runs file (can be .gz).",
    )
    parser.add_argument(
        "--canonical-elo",
        default=C.CANONICAL_LEADERBOARD_ELO_FILE,
        help="Path to the canonical ELO file (can be .gz).",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Automatically confirm merge without prompting.",
    )
    parser.add_argument(
        "--verbosity",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="DEBUG",
        help="Logging verbosity level.",
    )

    args = parser.parse_args()
    setup_merge_logging(args.verbosity)

    # --- Load Data -------------------------------------------------------
    logging.info(f"Loading local runs: {args.local_runs}")
    local_runs = load_json_file(args.local_runs)
    logging.info(f"Loading local ELO: {args.local_elo}")
    local_elo = load_json_file(args.local_elo)
    logging.info(f"Loading canonical runs: {args.canonical_runs}")
    canonical_runs = load_json_file(args.canonical_runs)
    logging.info(f"Loading canonical ELO: {args.canonical_elo}")
    canonical_elo = load_json_file(args.canonical_elo)

    if not all(isinstance(d, dict) for d in (local_runs, local_elo, canonical_runs, canonical_elo)):
        logging.error("Failed to load one or more data files correctly. Exiting.")
        sys.exit(1)

    # Ensure metadata dicts exist
    canonical_runs.setdefault("__metadata__", {})
    canonical_elo.setdefault("__metadata__", {})

    # --- Candidate Discovery --------------------------------------------
    candidates = find_merge_candidates(local_runs, local_elo, canonical_runs, canonical_elo)
    if not candidates:
        logging.info("No suitable candidates found for merging.")
        sys.exit(0)

    selected_candidates = select_candidates(candidates)
    if not selected_candidates:
        logging.info("No candidates selected for merging.")
        sys.exit(0)

    # --- Confirmation ----------------------------------------------------
    print("\n--- Summary of Merge ---")
    print(f"Models to merge: {', '.join(c['model_name'] for c in selected_candidates)}")
    print(f"Local Runs File:      {args.local_runs} (will be modified)")
    print(f"Local ELO File:       {args.local_elo} (will be modified)")
    print(f"Canonical Runs File:  {args.canonical_runs} (will be modified)")
    print(f"Canonical ELO File:   {args.canonical_elo} (will be modified)")
    print("------------------------")

    if not (
        args.yes
        or input("Proceed with merge? (y/n): ").strip().lower().startswith("y")
    ):
        logging.info("Merge cancelled by user.")
        sys.exit(0)

    # --- In-Memory Merge -------------------------------------------------
    logging.info("Starting merge process…")
    local_runs_copy = copy.deepcopy(local_runs)
    local_elo_copy = copy.deepcopy(local_elo)
    canonical_runs_copy = copy.deepcopy(canonical_runs)
    canonical_elo_copy = copy.deepcopy(canonical_elo)

    if not merge_data(
        selected_candidates,
        local_runs_copy,
        local_elo_copy,
        canonical_runs_copy,
        canonical_elo_copy,
    ):
        logging.error("No data moved during merge; aborting.")
        sys.exit(1)

    # --- Recalculate ELO -------------------------------------------------
    if not recalculate_canonical_elo(canonical_elo_copy):
        logging.error("ELO recalculation failed; no files will be written.")
        sys.exit(1)

    # --- Save Modified Files (transactional) --------------------------------
    logging.info("Saving modified files…")

    files_to_save = {
        args.canonical_runs: canonical_runs_copy,
        args.canonical_elo : canonical_elo_copy,
        args.local_runs    : local_runs_copy,
        args.local_elo     : local_elo_copy,
    }

    if _atomic_multi_save(files_to_save):
        logging.info("Merge process completed successfully.")
    else:
        logging.error("Merge aborted – no files were overwritten.")
        sys.exit(1)



if __name__ == "__main__":
    main()