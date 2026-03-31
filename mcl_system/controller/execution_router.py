import torch
import torch.nn as nn

class PolicyController(nn.Module):
    """Phase 6: Neural Decision Controller"""
    def __init__(self, lms_dim=256):
        super().__init__()
        # Outputs logits for 4 actions: FAST, REFLECT, DEEP, REFUSE
        self.decision_net = nn.Sequential(
            nn.Linear(lms_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        self.actions = ["FAST", "REFLECT", "DEEP", "REFUSE"]

    def forward(self, lms):
        logits = self.decision_net(lms)
        probs = torch.softmax(logits, dim=-1)
        action_idx = torch.argmax(probs, dim=-1).item()
        return self.actions[action_idx]

class ExecutionRouter:
    """Phase 7 & 8: Control Execution & Metacognitive Features"""
    def __init__(self, base_model, tokenizer):
        self.base = base_model
        self.tokenizer = tokenizer

    def execute(self, question, action, signals):
        if action == "REFUSE" or signals['error_prob'] > 0.85:
            return "I don't have enough confidence to answer this accurately.", 0
            
        elif action == "FAST":
            prompt = f"Q: {question}\nA: The answer is"
            return self._generate(prompt, max_tokens=15), 15
            
        elif action == "REFLECT":
            prompt = f"Q: {question}\nLet's consider if this is a trick question. Based on my familiarity, my confidence is {signals['confidence']:.2f}. The answer is"
            return self._generate(prompt, max_tokens=50), 50
            
        elif action == "DEEP":
            prompt = f"Q: {question}\nLet's think step-by-step to verify the logic:\n1."
            return self._generate(prompt, max_tokens=200), 200

    def _generate(self, prompt, max_tokens):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.base.device)
        outputs = self.base.model.generate(**inputs, max_new_tokens=max_tokens, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)