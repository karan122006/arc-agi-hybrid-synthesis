import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class BaseTransformerWrapper:
    """Phase 3 & 15: Upgraded Base Model Setup"""
    # Upgraded to Qwen 2.5 (0.5B) - A highly capable, modern reasoning model that runs well on CPU/Local setups.
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"): 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Upgraded Base LLM ({model_name}) on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Ensure padding token exists for the new model
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model (using float32 for CPU stability, float16 if you have a GPU)
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            output_hidden_states=True,
            torch_dtype=dtype
        ).to(self.device)

    def forward_pass(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.hidden_states[-1], outputs.logits, inputs['input_ids']