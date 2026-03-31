import pandas as pd
from datasets import load_dataset
import re

class DatasetHandler:
    def __init__(self):
        self.dataset = []

    def fetch_and_clean_data(self):
        """Phases 1 & 2: Fetching GSM8K and structuring to required format."""
        # Using HuggingFace as a reliable proxy for Kaggle benchmarks
        gsm8k = load_dataset("gsm8k", "main", split="train[:5000]") # Subset for speed
        
        cleaned_data = []
        for item in gsm8k:
            # Extract final answer from GSM8K format
            ans_match = re.search(r'####\s*(.*)', item['answer'])
            final_answer = ans_match.group(1).strip() if ans_match else None
            
            if not final_answer: continue
            
            # Determine heuristic difficulty based on reasoning steps (newlines)
            steps = len(item['answer'].split('\n'))
            difficulty = "hard" if steps > 4 else ("medium" if steps > 2 else "easy")
            
            cleaned_data.append({
                "question": self._normalize_text(item['question']),
                "answer": final_answer,
                "difficulty": difficulty,
                "type": "normal", # Can be augmented with deceptive/unanswerable logic
                "source": "gsm8k"
            })
            
        df = pd.DataFrame(cleaned_data)
        
        # Split (70/15/15)
        train_end = int(len(df) * 0.7)
        val_end = train_end + int(len(df) * 0.15)
        
        return {
            "train": df.iloc[:train_end],
            "val": df.iloc[train_end:val_end],
            "test": df.iloc[val_end:]
        }

    def _normalize_text(self, text):
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text