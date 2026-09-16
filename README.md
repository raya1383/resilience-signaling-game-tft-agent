Resilience Signaling Game: A TFT-Resilient Agent

An agent for a repeated incomplete-information signaling game, designed to study costly signaling, resilience, credibility, and strategic exit. The submitted policy detects Tit-for-Tat (TFT)-like behavior from the public action history and switches between role-aware response regimes.

Project overview

The game models strategic interaction between two players with private resilience and cost-management characteristics. Each round, a player selects a signaling intensity or exits. Continuing imposes effective costs, while signals may shape the opponent's belief about the player's type. The objective is to maximize game utility under incomplete information.

The final agent uses:

an initial low-cost probe;

public-history-based detection of repeated TFT-style imitation;

separate response cycles for the Iran and United States roles;

a fallback high-pressure regime after evidence contradicts TFT;

a submission checkpoint compatible with the course tournament interface.

Results

In a 10,000-game head-to-head evaluation against Tit-for-Tat:

Metric

Result

Agent mean utility

25.529

Tit-for-Tat mean utility

2.190

Mean margin

+23.339

95% confidence interval for margin

[21.236, 25.442]

In the full evaluation table, the agent achieved a mean score of 71.936, compared with 56.671 for Tit-for-Tat.

Repository layout

.
├── theory.py                 # Final policy and tournament integration code
├── submission.pt             # Final checkpoint for the tournament interface
├── evaluate.py               # Reproducible head-to-head evaluation script
├── requirements.txt          # Python dependencies used by the evaluation script
├── results/
│   ├── tft_evaluation.csv    # Per-game or aggregate evaluation outputs
│   └── summary.md            # Short description of the reported metrics
├── report/
│   ├── final_report.pdf      # Optional: final academic report
│   └── final_report.tex      # Optional: LaTeX source of the report
└── README.md

Do not upload the course-provided see/ engine unless its license and course rules explicitly allow redistribution. This repository contains only the project-specific policy, checkpoint, evaluation code, and documentation.

Running the evaluation

Place this repository beside an authorized local copy of the course game engine, install the required dependencies, and run the evaluator using the interface expected by your course release:

pip install -r requirements.txt
python evaluate.py

The exact command-line arguments may depend on the instructor-provided tournament package. theory.py and submission.pt are the files intended for the official submission workflow.

Notes on reproducibility

Reported results use matched seeds and evaluate both player roles.

Utility is the environment's raw game utility, not reward-shaping value.

The project-specific implementation does not modify the underlying game engine.

Results may differ slightly if the tournament package, seed set, or dependency versions differ.

Academic context

This project was developed for a game-theory and reinforcement-learning assignment. Its analysis considers Bayesian beliefs, costly signaling, effective continuation cost, strategic exit, and equilibrium implications alongside the computational evaluation.

License

Code written for this project is released under the MIT License. Third-party course materials and the game engine remain subject to their original terms.
