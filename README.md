# Certifying Language Model Rankings under Private and Malicious Feedback

GitHub repo for evaluating whether language model accuracy rankings can be certified from locally private reports containing a bounded fraction of malicious submissions.

The repo provides four complementary analyses:

| Analysis | Purpose | Script |
| --- | --- | --- |
| Paired evaluation | Evaluate models on shared labeled inputs and test ranking reliability under privacy and poisoning | `experiment_1.py` |
| Certificate efficiency | Compare a distributional confidence region with a direct accuracy-gap interval on identical reports | `experiment_2.py` |
| Adaptive robustness | Test report-adaptive attacks and sensitivity to the assumed malicious fraction | `experiment_3.py` |
| Linguistic subgroups | Examine subgroup rankings, report allocation, and simultaneous reliability | `experiment_4.py` |

## Contents

- [Installation](#installation)
- [Repository layout](#repository-layout)
- [Models and datasets](#models-and-datasets)
- [Usage](#usage)
- [Smoke test](#smoke-test)
- [Private and malicious paired evaluation](#private-and-malicious-paired-evaluation)
- [Certificate efficiency comparison](#certificate-efficiency-comparison)
- [Adaptive robustness and contamination-bound sensitivity](#adaptive-robustness-and-contamination-bound-sensitivity)
- [Linguistic subgroup analysis](#linguistic-subgroup-analysis)
- [Interpreting results](#interpreting-results)
- [Testing and reproducibility](#testing-and-reproducibility)

## Installation

Use Python 3.11. Create and activate a Conda environment, then install dependencies:

```bash
conda create -n naacl-ldp python=3.11
conda activate naacl-ldp
python -m pip install -r requirements.txt
```

The `requirements.txt` contains:

```text
ollama
datasets
numpy>=1.24
scipy>=1.10
pandas>=2.0
matplotlib>=3.7
tqdm>=4.65
```

Install and start Ollama separately before evaluating models. The Python package is the client; it does not install the Ollama server. Simulation, diagnostics, and plotting use cached results and do not require a running server.

## Repository layout

```text
experiment_1.py          # Model evaluation and private/poisoned ranking simulation
experiment_2.py         # Paired certificate efficiency comparison
experiment_3.py         # Adaptive attack and bound-sensitivity analysis
experiment_4.py         # Linguistic subgroup analysis
certificates.py        # Shared core imported by experiment3.py and experiment4.py
requirements.txt
README.md
```

Keep the scripts and `certificates.py` together. Use the shared core supplied with the subgroup package; the adaptive robustness package uses the same core. `experiment2.py` contains its own certificate routines.

## Models and datasets

| Task argument | Dataset | Split |
| --- | --- | --- |
| `snli` | SNLI (`stanfordnlp/snli`) | Test; examples without valid labels are excluded |
| `sst2` | SST-2 (`stanfordnlp/sst2`) | Validation |
| `boolq` | BoolQ (`aps/super_glue`, `boolq`) | Validation |

The commands below evaluate `qwen2.5:3b-instruct`, `qwen2.5:7b-instruct`, and `mistral:7b-instruct`. The script defaults are SNLI and the two Qwen models. Three models produce three model pairs per task, giving nine task/pair combinations across three tasks.

Both models answer the same labeled input. Categories are **A alone correct**, **B alone correct**, and **tie** (both correct or both incorrect). Malformed model outputs count as incorrect. The target gap is `accuracy(A) - accuracy(B)` over the cached examples.

All analyses use epsilon values **0.1, 0.5, 1, 2, 5** and the GRR and standard OUE channels. Smaller epsilon provides stronger category privacy. The default failure level is `delta=0.05`, with 1,000 simulation repetitions per setting; subgroup analysis allocates this failure budget across targets and methods.

## Usage


Run these commands from the repository root. Quote paths containing spaces. Full simulation grids can generate millions of trial rows; use the smoke test below to verify the setup first.

### Paired evaluation: evaluate once and simulate private reports

Pull all three models while the Ollama server is available:

```bash
ollama pull qwen2.5:3b-instruct
ollama pull qwen2.5:7b-instruct
ollama pull mistral:7b-instruct
```

Evaluate all three tasks/models, then simulate, and plot:

```bash
python experiment_1.py evaluate --tasks snli sst2 boolq --models qwen2.5:3b-instruct qwen2.5:7b-instruct mistral:7b-instruct --out three_models_three_tasks
python experiment_1.py simulate --tasks snli sst2 boolq --models qwen2.5:3b-instruct qwen2.5:7b-instruct mistral:7b-instruct --out three_models_three_tasks
python experiment_1.py plot --out three_models_three_tasks
```

### Certificate efficiency: compare certification methods

```bash
python experiment_2.py simulate --cache three_models_three_tasks/cache --tasks snli sst2 boolq --models qwen2.5:3b-instruct qwen2.5:7b-instruct mistral:7b-instruct --out experiment2_results
python experiment_2.py plot --csv experiment2_results/summary.csv --out experiment2_figures
```

### Adaptive robustness: stress-test attacks and contamination bounds

```bash
python experiment_3.py simulate --cache three_models_three_tasks/cache --tasks snli sst2 boolq --models qwen2.5:3b-instruct qwen2.5:7b-instruct mistral:7b-instruct --out experiment3_results
python experiment_3.py plot --csv experiment3_results/summary.csv --out experiment3_figures
```

### Linguistic subgroups: evaluate conditional rankings

```bash
python experiment_4.py simulate --cache three_models_three_tasks/cache --tasks snli sst2 boolq --models qwen2.5:3b-instruct qwen2.5:7b-instruct mistral:7b-instruct --out experiment4_results
python experiment_4.py plot --csv experiment4_results/summary.csv --out experiment4_figures
```
