import pandas as pd

class Evaluator:
    """Phase 9: Evaluation Metrics"""
    def __init__(self):
        self.results = []

    def log_result(self, is_correct, tokens_used, confidence, action):
        self.results.append({
            "correct": is_correct,
            "tokens": tokens_used,
            "confidence": confidence,
            "action": action
        })

    def compute_metrics(self):
        df = pd.DataFrame(self.results)
        if df.empty: return {}

        accuracy = df['correct'].mean()
        avg_tokens = df['tokens'].mean()
        
        # Brier Score for calibration: (confidence - actual_correct)^2
        calibration_error = ((df['confidence'] - df['correct'].astype(int))**2).mean()
        
        efficiency = accuracy / avg_tokens if avg_tokens > 0 else 0

        return {
            "Accuracy": round(accuracy, 4),
            "Calibration Error (Brier)": round(calibration_error, 4),
            "Avg Tokens": round(avg_tokens, 2),
            "Efficiency (Acc/Tokens)": round(efficiency, 6),
            "Action Distribution": df['action'].value_counts().to_dict()
        }