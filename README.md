# Capacity-Aware Multi-UE Handover Optimization for LEO Satellite Networks Using Dueling DDQN

This repository provides the source code, simulation configurations, baseline implementations, evaluation scripts, and data-processing utilities for the study:

**“Capacity-Aware Multi-UE Handover Optimization for LEO Satellite Networks Using Dueling DDQN.”**

The framework addresses multi-user handover in low Earth orbit (LEO) satellite networks under finite satellite capacity. Since satellite visibility and achievable link quality vary over time, handover decisions must jointly consider service availability, satellite occupancy, throughput, and connection stability.

The proposed framework uses a capacity-aware Dueling Double Deep Q-Network (DDQN) with sequential user equipment (UE) processing, shared satellite-occupancy tracking, feasible-action masking, link-quality information, remaining-capacity information, and connection-persistent association handling.

This public repository does **not** include pretrained model checkpoints, generated expert-policy files, evaluation CSVs, or paper result figures.

## Methods

### Reinforcement Learning Methods

| Method | Description | Source |
|---|---|---|
| Proposed | Capacity-aware Dueling DDQN with MSHBO expert-replay initialization | `rl/Dueling_ddqn.py` |
| Dueling DDQN (No-MSHBO) | Dueling DDQN without expert-replay initialization | `rl/Dueling_ddqn_no_mshbo.py` |
| Standard DDQN | DDQN using a standard non-dueling network architecture | `rl/Standard_ddqn.py` |
| Proposed w/o Remaining-Capacity State | Proposed framework with the remaining-capacity state component removed while retaining capacity constraints in the environment | `rl/Dueling_ddqn_no_capacity_state.py` |

### Non-RL Baselines

| Method | Source |
|---|---|
| MSHBO | `baselines/mshbo/MSHBO.cpp` |
| MSH | `baselines/msh/MSH.cpp` |
| GBW2 | `baselines/gbw2/GBW2.cpp` |
| MAC | `baselines/mac/MAC_capacity_fixed.cpp` |
| MVT | `baselines/mvt/MVT.cpp` |

The capacity-consistent MAC implementation is used as the primary MAC baseline. Historical implementations are retained under `baselines/reference_implementations/` for comparison and code inspection.

## Repository Structure

```text
.
├── rl/                  # RL models, training, and evaluation programs
├── baselines/           # Non-RL handover algorithms
├── configs/             # Centralized simulation, reward, scenario, and method settings
├── scripts/             # Unified experiment and plotting entry points
├── data_generation/     # Simulation-input processing utilities
├── plotting/            # Paper-figure and trade-off plotting scripts
├── tests/               # Unit and consistency tests
├── input.txt            # Satellite visibility intervals and scenario information
├── input_angle.txt      # Satellite/time-slot achievable-rate table
├── DATA_PROVENANCE.md   # Data provenance and redistribution notes
├── requirements.txt
└── README.md
```

Generated experiment outputs are written under `outputs/` and are excluded from version control through `.gitignore`.

## Centralized Configuration

The principal experimental settings are stored in `configs/`.

- `configs/simulation_config.json`  
  Defines input files, simulation duration, default UE/capacity settings, metric definitions, capacity accounting, and evaluation-order seeds.

- `configs/reward_config.json`  
  Defines the reward parameters used by the RL methods, including blocking, handover, throughput, persistence, short-term visibility, and rate normalization.

- `configs/experiment_scenarios.json`  
  Defines the UE-sensitivity, capacity-sensitivity, and capacity-state ablation scenarios.

- `configs/method_config.json`  
  Defines method-specific settings, including RL Top-K candidate selection, sticky probability, expert pretraining updates, random seeds, and the MAC minimum-stay parameter.

Explicit command-line arguments take precedence over the centralized JSON defaults.

The RL Top-K candidate-selection rule is applied only to the RL methods. Non-RL baselines retain their own original candidate-selection logic.

## Simulation Model

The experiments use satellite visibility, achievable-rate information, and shared multi-UE satellite-capacity constraints.

The formal evaluation interval contains

```text
3599 one-second time slots
```

with

```text
0 <= t < 3599.
```

Each satellite can simultaneously serve at most `C` UEs. Satellite occupancy is updated sequentially as UE association decisions are made, and active associations persist across consecutive slots unless visibility is lost, blocking occurs, or a handover releases the previous reservation.

### Input Files

#### `input.txt`

Contains the preprocessed satellite visibility information used by the simulator. The input satellite set contains satellites retained after visibility-based preprocessing rather than the complete original constellation.

Satellite IDs follow the zero-based input ordering used by the simulation code.

#### `input_angle.txt`

Contains the achievable-rate table in the format

```text
satellite_id time_slot achievable_rate_Mbps
```

The current public input contains the achievable-rate information used by the simulation.

Additional details on the origin and redistribution status of the simulation inputs are provided in:

```text
DATA_PROVENANCE.md
```

## Capacity-Aware RL Model

The proposed RL state includes:

- current satellite visibility,
- remaining satellite capacity,
- retained serving-satellite association,
- one-slot-ahead satellite visibility,
- normalized achievable rates, and
- normalized time.

The RL action space contains the satellite-selection actions and an explicit blocking action.

Capacity feasibility is maintained through:

1. shared satellite-occupancy tracking,
2. sequential UE processing,
3. connection-persistent reservations, and
4. feasible-action masking.

The currently retained serving satellite remains feasible at full occupancy because maintaining an existing association does not consume an additional capacity unit.

The default RL candidate-set size is:

```text
Top-K = 3
```

The blocking action is always retained.

## Reward Configuration

The default reward parameters are:

```text
Blocking penalty            lambda_b = 12
Handover penalty            lambda_h = 3
Throughput weight           lambda_t = 2
Connection-persistence bonus          = 0.05
Maximum look-ahead horizon            = 5
Future-visibility penalty             = 0.1
Rate normalization maximum            = 2946.289 Mbps
```

The same reward parameters are used by the learning-based comparison methods unless otherwise specified.

## Requirements

- Python 3.9 or later
- PyTorch
- NumPy
- pandas
- matplotlib
- Pillow
- A C++17-compatible compiler such as `g++`

A GPU is not required, although a CUDA-compatible PyTorch environment can accelerate training.

Install the Python dependencies from the repository root:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the virtual environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

## Compiling the Non-RL Baselines

Using `g++`:

```bash
mkdir build

g++ -std=c++17 -O2 baselines/mshbo/MSHBO.cpp \
    -o build/mshbo

g++ -std=c++17 -O2 baselines/msh/MSH.cpp \
    -o build/msh

g++ -std=c++17 -O2 baselines/gbw2/GBW2.cpp \
    -o build/gbw2

g++ -std=c++17 -O2 baselines/mac/MAC_capacity_fixed.cpp \
    -o build/mac_fixed

g++ -std=c++17 -O2 baselines/mvt/MVT.cpp \
    -o build/mvt
```

## Expert Policy Generation

The proposed method and the remaining-capacity-state ablation require an MSHBO expert-policy file for replay-buffer initialization.

Generate the MSHBO policy with:

```bash
build/mshbo < input.txt
```

The resulting expert-policy CSV is then supplied to the RL training program.

Expert trajectories are used only for replay-buffer initialization and pretraining. They do not constrain the final learned policy.

## Training

### Proposed Capacity-Aware Dueling DDQN

```bash
python rl/Dueling_ddqn.py \
    --input input.txt \
    --input_angle input_angle.txt \
    --mshbo_csv policy.csv \
    --episodes 300 \
    --seed 42 \
    --eval_order_seed 42 \
    --output_dir runs/proposed_seed42
```

### Dueling DDQN without MSHBO Initialization

```bash
python rl/Dueling_ddqn_no_mshbo.py \
    --input input.txt \
    --input_angle input_angle.txt \
    --episodes 300 \
    --seed 42 \
    --eval_order_seed 42 \
    --output_dir runs/no_mshbo_seed42
```

### Standard DDQN

```bash
python rl/Standard_ddqn.py \
    --input input.txt \
    --input_angle input_angle.txt \
    --episodes 300 \
    --seed 42 \
    --eval_order_seed 42 \
    --output_dir runs/standard_seed42
```

### Proposed without Remaining-Capacity State

```bash
python rl/Dueling_ddqn_no_capacity_state.py \
    --input input.txt \
    --input_angle input_angle.txt \
    --mshbo_csv policy.csv \
    --episodes 300 \
    --seed 42 \
    --eval_order_seed 42 \
    --output_dir runs/no_capacity_state_seed42
```

The formal default configuration uses:

```text
Training episodes             300
Learning rate                 1e-3
Discount factor               0.99
Mini-batch size               256
Replay-buffer capacity        200000
Target-network update         2000 global time slots
Epsilon                       1.0 -> 0.05
Epsilon decay constant        30000
Top-K                         3
Sticky probability            0.2
Expert pretraining updates    2000
```

## Evaluation

Unified experiment entry points are provided under `scripts/`.

Without `--execute`, the scripts validate the required inputs and checkpoints without running the full evaluation.

### UE Sensitivity

```bash
python scripts/run_ue_sensitivity.py
```

Run the complete evaluation with:

```bash
python scripts/run_ue_sensitivity.py --execute
```

### Satellite-Capacity Sensitivity

```bash
python scripts/run_capacity_sensitivity.py
```

Run the complete evaluation with:

```bash
python scripts/run_capacity_sensitivity.py --execute
```

### Capacity-State Ablation

```bash
python scripts/run_ablation.py
```

Run the complete evaluation with:

```bash
python scripts/run_ablation.py --execute
```

The principal evaluation programs are:

- `rl/evaluate_rl_multiseed_lambda3.py` — UE-sensitivity evaluation
- `rl/evaluate_capacity_sensitivity.py` — satellite-capacity sensitivity
- `rl/evaluate_capacity_ablation_vs_ue.py` — remaining-capacity-state ablation
- `rl/evaluate_rl_fulllegal_multiseed.py` — UE-processing-order sensitivity

## Experiment Scenarios

### UE Sensitivity

```text
UE = {10, 15, 20, 25, 30}
Satellite capacity C = 5
```

### Satellite-Capacity Sensitivity

```text
UE = 25
C = {1, 3, 5, 7, 9}
```

### Evaluation-Order Seeds

```text
42, 123, 2024, 2025, 2026
```

These seeds are used to evaluate sensitivity to the sequential UE-processing order under a fixed trained checkpoint. They **do not represent independently trained RL models**.

## Performance Metrics

The principal metrics are:

- time-average system throughput,
- average blocking rate, and
- average number of satellite-to-satellite handovers per UE.

A blocked UE contributes zero system throughput.

Transitions between a satellite and the blocking state are treated as service interruption or reconnection and are not counted as successful satellite-to-satellite handovers.

## Plotting

Paper-figure generation utilities are provided under `plotting/`.

After the required evaluation outputs have been generated, validate the plotting inputs with:

```bash
python scripts/plot_paper_figures.py
```

Generate the figures with:

```bash
python scripts/plot_paper_figures.py --execute
```

The plotting entry point validates the required CSV schema before generating figures and stops explicitly when required evaluation data are unavailable.

Additional plotting utilities, including the blocking-handover trade-off visualization, are also provided under `plotting/`.

## Testing

Run the repository tests with:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

A lightweight Python syntax check can also be performed with:

```bash
python -m compileall -q rl scripts plotting tests
```

## Reproducibility Scope

This repository provides the source implementation, centralized configuration files, simulation inputs, experiment entry points, baseline implementations, and plotting utilities used for the study.

Pretrained RL checkpoints, generated expert-policy files, evaluation CSVs, and generated result figures are intentionally not included. Therefore, reproducing the complete numerical results requires training the corresponding RL models, generating the required baseline policies, and running the evaluation pipeline.

The evaluation-order experiments use multiple UE-processing-order seeds with fixed trained checkpoints. They should not be interpreted as independent training runs.

Some historical baseline implementations are retained only for reference and may use capacity-accounting or metric conventions that differ from the current common evaluation framework. The primary implementations listed in the **Methods** section should be used for formal comparisons.

Please also consult `DATA_PROVENANCE.md` before redistributing or replacing the provided satellite-derived simulation inputs.

## Citation

If you use this repository in academic work, please cite the associated paper:

```bibtex
@inproceedings{chou2027capacityaware,
  title   = {Capacity-Aware Multi-UE Handover Optimization for LEO Satellite Networks Using Dueling DDQN},
  author  = {Chou, Po-Heng and Wang, Chiapin and Huang, Chung-Chi and Chen, Kuan-Hao and Chen, Chien-Yu},
  note    = {Manuscript},
  year    = {2027}
}
```

The citation information will be updated after publication.

## Contact

For questions regarding the implementation or experimental setup, please open an issue in this repository.
