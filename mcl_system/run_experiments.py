import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data.dataset_handler import DatasetHandler
from main import MetacognitiveSystem

class StaticPolicy:
    """A dummy policy to simulate a standard, non-metacognitive LLM"""
    def __init__(self, action):
        self.action = action
        self.actions = ["FAST", "REFLECT", "DEEP", "REFUSE"]
    
    def __call__(self, lms):
        return self.action

def run_evaluation():
    print("Loading Dataset for Experiments (50 Samples)...")
    handler = DatasetHandler()
    splits = handler.fetch_and_clean_data()
    test_data = splits['test'].head(50) 
    
    system = MetacognitiveSystem()
    system.load_weights()
    
    # Save the smart, trained policy
    trained_policy = system.policy
    
    # ==========================================
    # 1. RUN BASELINE (Static 'DEEP' Reasoning)
    # ==========================================
    print("\n[1/2] Running BASELINE (Standard Static LLM - Always DEEP)...")
    system.policy = StaticPolicy("DEEP") # Force it to act like a standard CoT model
    system.evaluator.results = [] # Clear metrics
    
    baseline_results = []
    for idx, row in test_data.iterrows():
        result = system.process_query(row['question'], row['answer'], training=False)
        baseline_results.append(result)
        
    baseline_metrics = system.evaluator.compute_metrics()
    baseline_df = pd.DataFrame(system.evaluator.results)
    
    # ==========================================
    # 2. RUN MCL SYSTEM (Adaptive SST-MSV)
    # ==========================================
    print("\n[2/2] Running METACOGNITIVE SYSTEM (Adaptive Routing)...")
    system.policy = trained_policy # Restore the trained brain
    system.evaluator.results = [] # Clear metrics
    
    mcl_results = []
    for idx, row in test_data.iterrows():
        result = system.process_query(row['question'], row['answer'], training=False)
        mcl_results.append(result)
        
    mcl_metrics = system.evaluator.compute_metrics()
    mcl_df = pd.DataFrame(system.evaluator.results)

    print("\n=== EXPERIMENT COMPLETE ===")
    generate_plots(baseline_metrics, mcl_metrics, mcl_df)

def generate_plots(baseline_metrics, mcl_metrics, mcl_df):
    """Generates the Phase 10 Research Visualizations"""
    sns.set_theme(style="whitegrid")
    
    # PLOT 1: Baseline vs MCL Comparison (Accuracy & Efficiency)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('SST-MSV Architecture vs Baseline LLM', fontsize=16)
    
    models = ['Baseline (Static)', 'MCL (Adaptive)']
    
    # Accuracy Comparison
    acc_data = [baseline_metrics.get('Accuracy', 0), mcl_metrics.get('Accuracy', 0)]
    sns.barplot(x=models, y=acc_data, ax=axes[0], palette="Blues")
    axes[0].set_title('Accuracy')
    axes[0].set_ylabel('Win Rate')
    
    # Efficiency Comparison
    eff_data = [baseline_metrics.get('Efficiency (Acc/Tokens)', 0), mcl_metrics.get('Efficiency (Acc/Tokens)', 0)]
    sns.barplot(x=models, y=eff_data, ax=axes[1], palette="Greens")
    axes[1].set_title('Compute Efficiency (Accuracy / Token Cost)')
    axes[1].set_ylabel('Efficiency Score')
    
    plt.tight_layout()
    plt.savefig('experiment_1_comparison.png')
    print("Saved -> experiment_1_comparison.png")
    
    # PLOT 2: Token Usage vs Policy Decisions
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=mcl_df, x='action', y='tokens', palette="Set2", order=["FAST", "REFLECT", "DEEP", "REFUSE"])
    plt.title('Token Expenditure by Metacognitive Action')
    plt.xlabel('Chosen Action')
    plt.ylabel('Tokens Used')
    plt.savefig('experiment_2_tokens_vs_action.png')
    print("Saved -> experiment_2_tokens_vs_action.png")

    # PLOT 3: Calibration Curve (Confidence vs Actual Correctness)
    plt.figure(figsize=(8, 5))
    # Bin the confidence scores
    mcl_df['conf_bin'] = pd.cut(mcl_df['confidence'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[0.1, 0.3, 0.5, 0.7, 0.9])
    calibration_data = mcl_df.groupby('conf_bin', observed=False)['correct'].mean().reset_index()
    
    sns.lineplot(data=calibration_data, x='conf_bin', y='correct', marker='o', linewidth=2, color='purple')
    plt.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration") # Diagonal line
    
    plt.title(f"Metacognitive Calibration Curve (Brier: {mcl_metrics.get('Calibration Error (Brier)', 0):.4f})")
    plt.xlabel('Predicted Confidence (LMS Head)')
    plt.ylabel('Actual Accuracy')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig('experiment_3_calibration.png')
    print("Saved -> experiment_3_calibration.png")
    
    print("\nAll experiment visualizations have been generated successfully in your project folder!")

if __name__ == "__main__":
    run_evaluation()