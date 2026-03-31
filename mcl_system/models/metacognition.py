import torch
import torch.nn as nn

class StateStreamTransformer(nn.Module):
    """Phase 4 & 5: State Stream and LMS Signals"""
    def __init__(self, hidden_dim=768, lms_dim=256):
        super().__init__()
        
        self.cnn = nn.Conv1d(in_channels=hidden_dim, out_channels=128, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=128, hidden_size=lms_dim, batch_first=True)
        
        self.mlp = nn.Sequential(
            nn.Linear(lms_dim, lms_dim),
            nn.GELU(),
            nn.Linear(lms_dim, lms_dim)
        )
        
        self.confidence_head = nn.Linear(lms_dim, 1)
        self.uncertainty_head = nn.Linear(lms_dim, 1)
        self.error_prob_head = nn.Linear(lms_dim, 1) 

    def forward(self, hidden_states):
        x = hidden_states.transpose(1, 2)
        x = torch.relu(self.cnn(x))
        x = x.transpose(1, 2)
        
        lstm_out, _ = self.lstm(x)
        global_state = lstm_out[:, -1, :] 
        lms = self.mlp(global_state)
        
        confidence = torch.sigmoid(self.confidence_head(lms))
        uncertainty = torch.nn.functional.softplus(self.uncertainty_head(lms))
        error_prob = torch.sigmoid(self.error_prob_head(lms))
        
        # REMOVED .item() SO GRADIENTS CAN FLOW DURING TRAINING
        return {
            "lms": lms,
            "confidence": confidence.squeeze(),
            "uncertainty": uncertainty.squeeze(),
            "error_prob": error_prob.squeeze()
        }