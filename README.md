# EQ-Bench 3

EQ-Bench 3 is a multi-turn emotional intelligence benchmark. It assesses active EQ skills, interpersonal skills, psychological insight and analytical depth. It challenges language models with role-play or analysis tasks that require empathy, depth of insight, social dexterity, and more. An auxiliary judge model (by default, Claude Sonnet 3.7) scores or pairwise-compares the outputs.

For full details on the benchmark including methodology, criteria, bias analysis, repeatabilty experiments and more, click [here](http://localhost:8000/about.html#long-version-header).

**Features**  
- **Role-Play Scenarios**: The tested model is placed in conversation-based scenarios (e.g., parenting, relationship conflict, workplace tension). It must articulate what it (and others) feel/think before delivering its final response.  
- **Analysis Scenarios**: The model is asked to read or interpret a transcript and perform an in-depth analysis of human dynamics and subtext.  
- **Rubric Scoring**: A judge LLM assigns a multi-criteria rubric score (0–20, scaled up to 0–100) to each scenario outcome, focusing on empathy, emotional reasoning, social skill, etc.  
- **Pairwise ELO Analysis**: The judge compares transcripts from two different models for the same scenario, awarding a “win” margin for each of several criteria. A TrueSkill/ELO-like solver aggregates all pairwise comparisons to produce a final ranking.

## Leaderboard

The full EQBench3 leaderboard is viewable at: [https://eqbench.com/](https://eqbench.com/)

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Installation & Requirements](#installation--requirements)  
3. [Quickstart](#quickstart)  
4. [Running the Benchmark](#running-the-benchmark)  
   - [Command-Line Arguments](#command-line-arguments)  
   - [Example Commands](#example-commands)  
   - [Rubric vs ELO](#rubric-vs-elo)  
   - [Local vs Canonical Results Files](#local-vs-canonical-results-files)  
5. [Merging Local Results into the Canonical Leaderboard](#merging-local-results-into-the-canonical-leaderboard)  
6. [Folder and File Structure](#folder-and-file-structure)  
7. [Limitations and Notes](#limitations-and-notes)  
8. [Contact and License](#contact-and-license)  
9. [Citation](#citation)

---

## 1. Project Overview

EQ-Bench 3 aims to measure “active emotional intelligence” abilities in LLMs. Rather than knowledge-based or short-answer questions, tasks here are multi-turn dialogues or analysis questions that test empathy, social dexterity, and psychological insight. The tested model’s responses are then evaluated by a judge model:
- **Rubric pass**: The judge model issues a numerical score for each scenario.  
- **ELO pass**: The judge model performs pairwise comparisons of transcripts from different models, resulting in an overall ELO ranking (via TrueSkill).

### Key Points
- **Scenarios** vary from relationship drama to conflict mediation, pushing the tested model to reason about others’ emotions.  
- **Analysis tasks** require deeper reflection on a provided transcript or scenario.  
- **Judge model**: By default, a Claude model (Sonnet 3.7) is used, but any LLM accessible via your environment (like GPT-4.1) could serve as judge.  
- **Truncation**: Pairwise judgments truncate outputs to level the playing field. Rubric judgments typically do not truncate (to preserve detail).  

---

## 2. Installation & Requirements

1. Clone this repository:

   ```bash
   git clone https://github.com/EQ-bench/eqbench3.git
   cd eqbench3
   ```

2. (Optional) Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure your API keys in `.env`:

   ```bash
   cp .env.example .env
   # then edit .env with your API keys, e.g.:
   # TEST_API_KEY=sk-...
   # JUDGE_API_KEY=sk-...
   ```

   - `TEST_API_KEY` & `TEST_API_URL` are used when calling the tested model.  
   - `JUDGE_API_KEY` & `JUDGE_API_URL` are used by the judge model.  

---

## 3. Quickstart

1. **Run a single iteration** of the entire EQ-Bench (both scenario simulation and scoring) for a model:
   ```bash
   python eqbench3.py \
     --test-model openai/gpt-4.1-mini \
     --judge-model anthropic/claude-3.7-sonnet \
     --no-elo \
     --no-rubric
   ```
   This will simply run each scenario but skip ELO and Rubric steps. The partial transcripts appear in `eqbench3_runs.json`.

2. **Enable Rubric Scoring**:
   ```bash
   python eqbench3.py \
     --test-model openai/gpt-4.1-mini \
     --model-name gpt-4.1-mini-demo-run \
     --judge-model anthropic/claude-3.7-sonnet \
     --no-elo \
     --iterations 2
   ```
   - Runs 2 iterations of every scenario, scoring them with the rubric.  
   - Data is recorded in `eqbench3_runs.json` with final average rubric in `.results.average_rubric_score`.

3. **Enable Full ELO**:
   ```bash
   python eqbench3.py \
     --test-model openai/gpt-4.1-mini \
     --model-name my-gpt4-run \
     --judge-model anthropic/claude-3.7-sonnet
   ```
   - After scenario simulation, it does a multi-stage pairwise pass (comparing `my-gpt4-run` to known models in local+leaderboard data).  
   - A final ELO rating is stored in `elo_results_eqbench3.json` (and will appear in `.results.elo_raw` in `eqbench3_runs.json`).

---

## 4. Running the Benchmark

You interact with the main script [`eqbench3.py`](./eqbench3.py). It orchestrates:
- **Scenario simulation** (multi-turn, plus optional debrief).  
- **Rubric scoring** (judge LLM reads final transcripts and issues a 0–100 scaled rating).  
- **ELO analysis** (judge LLM does pairwise comparisons, then solves rating with TrueSkill).  

### Command-Line Arguments

| Argument                           | Description                                                                                                                             |
|------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| `--test-model` **(required)**      | API model identifier for the tested model (e.g. `openai/gpt-4.1-mini`).                                                                       |
| `--model-name`                     | Logical name used for storing ELO data (defaults to `--test-model` if not supplied). It must be unique to avoid collisions in ELO data. |
| `--judge-model`                    | Model to be used as the judge for ELO and/or rubric scoring. If `--no-elo` and `--no-rubric` are both set, you can skip this.           |
| `--runs-file`                      | Local runs data file (`.json` or `.json.gz`) storing scenario transcripts & statuses. Default: `eqbench3_runs.json`.                   |
| `--elo-results-file`               | Local ELO data file for storing pairwise comparisons & final ELO ratings. Default: `elo_results_eqbench3.json`.                        |
| `--leaderboard-runs-file`          | Path to the canonical leaderboard runs data (read-only). Default: `data/canonical_leaderboard_results.json.gz`.                         |
| `--leaderboard-elo-file`           | Path to the canonical leaderboard ELO data (read-only). Default: `data/canonical_leaderboard_elo_results.json.gz`.                     |
| `--run-id`                         | Custom prefix for the run key. If omitted, a random hex is used.                                                                        |
| `--threads`                        | Number of parallel threads (for scenario generation or scoring).                                                                        |
| `--verbosity`                      | Logging level: `DEBUG`, `INFO`, `WARNING`, etc. Default is `INFO`.                                                                      |
| `--save-interval`                  | Save progress every N tasks. Default is 2.                                                                                             |
| `--iterations`                     | Number of times to run each scenario. Default is 1.                                                                                     |
| `--no-elo`                         | If set, skip the ELO analysis step.                                                                                                      |
| `--no-rubric`                      | If set, skip the rubric scoring step.                                                                                                   |
| `--redo-rubric-judging`            | Reset tasks’ rubric status so that rubric scoring is done again.                                                                        |
| `--reset-model`                    | Wipes all local data (runs & ELO comparisons) pertaining to `--model-name` before starting.                                             |

Run `python eqbench3.py --help` for the full usage.

### Example Commands

1. **Rubric-Only**:  
   ```bash
   python eqbench3.py \
     --test-model "openai/gpt-4.1" \
     --judge-model "anthropic/claude-3.7-sonnet" \
     --no-elo \
     --iterations 3
   ```
   - Runs each scenario 3 times, finishing with a final rubric score in `eqbench3_runs.json`.

2. **Full ELO**:  
   ```bash
   python eqbench3.py \
     --test-model "anthropic/claude-3.7-sonnet" \
     --model-name "claude-3.7-expt1" \
     --judge-model "anthropic/claude-3.7-sonnet"
   ```
   - Compares `claude-3.7-expt1` to other data in `elo_results_eqbench3.json` + `data/canonical_leaderboard_elo_results.json.gz`, updating local ELO.

3. **Force Overwrite** (If you want to re-run rubric scroing on a previously finished run):
   ```bash
   python eqbench3.py \
     --test-model "openai/gpt-4.1-mini" \
     --judge-model "anthropic/claude-3.7-sonnet" \
     --redo-rubric-judging \
     --no-elo
   ```

### Rubric vs ELO
- **Rubric**:  
  - Each scenario is scored individually on a 0–20 scale (scaled to 0–100 in summary).  
  - Less variance in cost; no direct competition among models.  
  - Good if you want an absolute measure for a single model, or when you only have one model to test.

- **ELO** (TrueSkill)  
  - The judge compares transcripts from different models for each scenario, awarding a margin-based “win.”  
  - Requires at least one baseline or previously completed model to compare against.  
  - It can be more discriminative at the top, but also more expensive if you have many existing models.  
  - Typically requires final results to be saved in `elo_results_eqbench3.json` (local) or the canonical `data/canonical_leaderboard_elo_results.json.gz`.

**You can run either or both**. If you do both, you end up with a final ELO rating as well as an average rubric score.

### Local vs Canonical Results Files
- **Local Files**:  
  - `eqbench3_runs.json`: Stores detailed transcripts, partial results, scenario statuses.  
  - `elo_results_eqbench3.json`: Stores local pairwise comparisons plus your local ELO ratings for each model.  
  - Safe to commit or ignore, depending on your usage.  
- **Canonical Files**:  
  - `data/canonical_leaderboard_results.json.gz`  
  - `data/canonical_leaderboard_elo_results.json.gz`  
  - These represent the official leaderboard data. If you run ELO with these files, your model is compared to previously included models from the central scoreboard.  

**Why specify your own canonical results files?**  
- If you don’t want to compare your model to the entire official scoreboard, you can point to a different `--leaderboard-runs-file` and `--leaderboard-elo-file`. This could be a partial or private scoreboard.

---

## 5. Merging Local Results into the Canonical Leaderboard

Once you’re satisfied with your local run and want to merge your new model’s data into a canonical leaderboard:
1. **Review** your local results in `eqbench3_runs.json` and `elo_results_eqbench3.json`.
2. **Use** `merge_results_to_canonical.py` to push selected runs from local to canonical files:
   ```bash
   python merge_results_to_canonical.py \
     --local-runs eqbench3_runs.json \
     --local-elo elo_results_eqbench3.json \
     --canonical-runs data/canonical_leaderboard_results.json.gz \
     --canonical-elo data/canonical_leaderboard_elo_results.json.gz
   ```
   - This script interactively shows which runs can be merged and asks for confirmation.  
   - After merging, it recalculates the canonical ELO and updates canonical data transactionally.

3. **The newly integrated model** then appears in `data/canonical_leaderboard_elo_results.json.gz` with a final rating.

---

## 6. Folder and File Structure

- **`eqbench3.py`**: Main entry point for generating scenario outcomes, running rubric scoring, and orchestrating ELO.  
- **`merge_results_to_canonical.py`**: Script that merges local runs/ELO data with canonical results and re-runs an ELO solve.  
- **`data/`**: Contains canonical leaderboard data (`.json.gz`), scenario prompts, and supporting prompt templates.  
- **`utils/`**: Helper modules for file I/O, logging, API calls.  
- **`core/`**: Contains the main logic for scenario simulation, ELO, TrueSkill solver, pairwise judgement, and other components.  
- **`eqbench3_runs.json`** (local default): Your local scenario transcripts & statuses.  
- **`elo_results_eqbench3.json`** (local default): Your local ELO comparisons & ratings.  
- **`.env`**: Environment file specifying API credentials & certain timeouts.

---

## 7. Limitations and Notes

1. **Subjectivity**. Scores rely on a single LLM judge, which may exhibit biases or partialities.  
2. **Truncation**. Pairwise judgments deliberately truncate outputs to mitigate length bias.  
3. **Comparisons**. ELO is only as reliable as the coverage of pairwise matchups. With fewer references, the final rating is less stable.  
4. **Costs**. Each scenario run plus judge calls can be nontrivial in API usage. For large LLMs or many scenarios, expect higher costs.  
5. **Interpretation**. “EQ” is a broad concept. This benchmark does not aim for an absolute definition—only a measure of how well the model responds to intricate social/emotional scenarios under a consistent LLM judge.

---

## 8. Contact and License

- **License**: This project is open-sourced under an [MIT license](./LICENSE)
- **Contact**: For questions, refer to the [Issues](https://github.com/EQ-bench/eqbench3/issues) section of this repository.  

Contributions or suggestions are welcome—please open an issue or pull request.

## 9. Citation

If you use EQ-Bench 3 in academic work, please cite both the benchmark
repository and the original EQ-Bench paper.

```bibtex
# Benchmark repository (update the version / commit as appropriate)
@misc{eqbench3_repo_2025,
  author       = {Samuel J. Paech},
  title        = {EQ-Bench 3: Emotional Intelligence Benchmark},
  year         = {2025},
  howpublished = {\url{https://github.com/EQ-bench/eqbench3}},
  note         = {Commit \<hash\> or release \<tag\>}
}

@misc{paech2023eqbench,
  title        = {EQ-Bench: An Emotional Intelligence Benchmark for Large Language Models},
  author       = {Samuel J. Paech},
  year         = {2023},
  eprint       = {2312.06281},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL}
}
