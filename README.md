#Hybrid Neural-Symbolic Architecture for ARC-AGI
This repository contains a standalone implementation of a hybrid neural-symbolic solver developed for the Kaggle Measuring AGI (ARC Prize) competition. The architecture explores the intersection of deep perceptual embeddings and discrete algorithmic search to solve the Abstraction and Reasoning Corpus (ARC) tasks.
#🔬 Architecture & Technical MethodologyThe core philosophy of this solver is that AGI-adjacent tasks require two distinct systems: System 1 (Perception) for pattern recognition and System 2 (Reasoning) for logical rule induction.
1. Perception Layer: Multi-Modal TransformerWe utilize a transformer-based encoder to map 2D grid structures into a high-dimensional latent space.Feature Extraction: Grids are treated as spatial sequences. We use specialized positional encodings to maintain 2D topological relationships.Augmentation: The model utilizes $D_4$ symmetry groups (rotations and reflections) during the embedding phase to ensure geometric invariance across diverse task orientations.
2. Representation: Graph Neural-Symbolic ProcessorInstead of raw pixel manipulation, the system represents tasks as Dynamic Knowledge Graphs.Object Detection: Connected Component Labeling (CCL) and color-based segmentation identify discrete "entities."Relational Mapping: A Graph Neural Network (GNN) computes edge weights between entities based on spatial proximity, color similarity, and shape congruence.Embedding: Visual features and symbolic relations are fused into a unified Rule Embedding vector used to guide the subsequent search.
3. Logic Engine: DSL & Program SynthesisThe solver synthesizes programs in a custom Domain-Specific Language (DSL) optimized for high-dimensional search efficiency.Guided Search: We employ a Prioritized Best-First Search where the search path is pruned and guided by the neural embeddings from the GNN.DSL Primitives: Includes robust operations for crop, move, recolor, reflect, and fill_pattern.Optimization: To avoid combinatorial explosion, we use Experience Replay Memory to prioritize primitives that have historically solved tasks with similar graph embeddings.
4. Stability: Cross-Validation & Voting EnsembleTo ensure generalization on few-shot examples (3-5 grids), we implement a multi-stage validation pipeline:Candidate Generation: The synthesis engine produces the top-K candidate programs.Validation: Each program is executed on the training inputs; only programs with 100% training accuracy are passed to the ensemble.Selection: If multiple programs pass, the one with the lowest Kolmogorov complexity (shortest DSL length) is selected as the optimal prediction.
#🛠️ Developer Setup
PrerequisitesPython 3.10+
PyTorch 2.0+ (CUDA recommended)
NetworkX (for graph processing)
#Installation
git clone https://github.com/[Your-Username]/arc-agi-hybrid-solver.git
cd arc-agi-hybrid-solver
pip install -r requirements.txt
#Running the Solver
To execute the solver on a specific task:
python run_solver.py --task_id [TASK_ID] --use_gpu --visualize
📜 LicenseDistributed under the Apache License 2.0. See LICENSE for more information.
