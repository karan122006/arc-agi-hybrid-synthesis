import os
import json
import torch
import torch.optim as optim
from data.dataset_handler import DatasetHandler
from models.base_llm import BaseTransformerWrapper
from models.metacognition import StateStreamTransformer
from controller.execution_router import PolicyController, ExecutionRouter
from evaluation.metrics import Evaluator

class MetacognitiveSystem:
    def __init__(self):
        print("Initializing Base LLM...")
        self.base_wrapper = BaseTransformerWrapper() 
        
        print("Initializing State Stream & Controller...")
        self.state_stream = StateStreamTransformer(hidden_dim=self.base_wrapper.model.config.hidden_size) 
        self.policy = PolicyController(lms_dim=256)
        self.router = ExecutionRouter(self.base_wrapper, self.base_wrapper.tokenizer)
        self.evaluator = Evaluator()
        
        self.optimizer = optim.Adam(self.state_stream.parameters(), lr=0.001)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=0.005)

    def load_weights(self, path="mcl_weights.pth"):
        """Load pre-trained metacognitive brains if they exist"""
        if os.path.exists(path):
            print(f"Loading trained weights from {path}...")
            checkpoint = torch.load(path, map_location=self.base_wrapper.device, weights_only=True)
            self.state_stream.load_state_dict(checkpoint['state_stream_state_dict'])
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        else:
            print("No saved weights found, starting fresh with a clean slate.")

    def save_weights(self, path="mcl_weights.pth"):
        """Save the trained metacognitive brains to disk"""
        print(f"\nSaving Metacognitive weights to {path}...")
        torch.save({
            'state_stream_state_dict': self.state_stream.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
        }, path)

    def train_step(self, question, is_correct):
        """Supervised learning for Confidence/Error prediction"""
        hidden_states, _, _ = self.base_wrapper.forward_pass(question)
        signals = self.state_stream(hidden_states)
        
        target_confidence = torch.tensor(1.0 if is_correct else 0.0).to(self.base_wrapper.device)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(signals['confidence'], target_confidence)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train_policy(self, lms, action, is_correct, tokens_used):
        """Reinforcement Learning to optimize the Action Router"""
        lambda_penalty = 0.005 
        accuracy_reward = 1.0 if is_correct else -0.5
        reward = accuracy_reward - (lambda_penalty * tokens_used)
        
        action_idx = self.policy.actions.index(action)
        logits = self.policy.decision_net(lms)
        probs = torch.softmax(logits, dim=-1)

        log_prob = torch.log(probs[0, action_idx] + 1e-8) 
        policy_loss = -log_prob * reward 
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return reward, policy_loss.item()

    def process_query(self, question, expected_answer=None, training=False):
        hidden_states, _, _ = self.base_wrapper.forward_pass(question)
        signals = self.state_stream(hidden_states)
        action = self.policy(signals['lms'])
        final_answer, tokens_used = self.router.execute(question, action, signals)
        
        is_correct = False
        if expected_answer:
            import re
            # Extract all numbers from the model's answer
            found_numbers = re.findall(r'\d+', final_answer)
            # If the exact expected number is in the output, count it as correct!
            if expected_answer in found_numbers:
                is_correct = True
            
        if training:
            state_loss = self.train_step(question, is_correct)
            with torch.no_grad():
               updated_signals = self.state_stream(hidden_states)
            reward, pol_loss = self.train_policy(updated_signals['lms'], action, is_correct, tokens_used)
            # Only print training info if we are actually training
            print(f"  -> State Loss: {state_loss:.4f} | RL Reward: {reward:.4f}")
            
        conf_val = signals['confidence'].item()
        self.evaluator.log_result(is_correct, tokens_used, conf_val, action)

        return {
            "answer": final_answer.strip()[:150] + "...", 
            "confidence": round(conf_val, 4),
            "decision": action
        }

if __name__ == "__main__":
    print("Loading Dataset...")
    handler = DatasetHandler()
    splits = handler.fetch_and_clean_data()
    test_data = splits['test'].head(20) 
    
    system = MetacognitiveSystem()
    
    # PHASE 15: Load previous memories (if they exist)
    system.load_weights()
    
    print("\n--- Running Upgraded Metacognitive Pipeline ---")
    for idx, row in test_data.iterrows():
        print(f"\nQuery: {row['question'][:60]}...")
        result = system.process_query(row['question'], row['answer'], training=False) 
        print(json.dumps(result, indent=2))
        
    print("\n--- Final Evaluation Metrics ---")
    metrics = system.evaluator.compute_metrics()
    print(json.dumps(metrics, indent=2))
    
    # PHASE 15: Save new memories to disk
    system.save_weights()