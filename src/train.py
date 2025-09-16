import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class VIQN(nn.Module):
    def __init__(self, hidden, K, tau_embed=32):
        super().__init__()
        self.tau_mlp = nn.Sequential(nn.Linear(1, tau_embed), nn.SiLU())
        self.out = nn.Linear(hidden + tau_embed, K)
        
    def forward(self, h, taus):
        te = self.tau_mlp(taus)
        h = h.unsqueeze(1).expand(-1, taus.size(1), -1)
        x = torch.cat([h, te], -1)
        return self.out(x)

class MetaRiskAdapter(nn.Module):
    def __init__(self, embed_dim, K):
        super().__init__()
        self.hyper_net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, K),
            nn.Sigmoid()
        )
        
    def forward(self, prompt_embedding):
        return self.hyper_net(prompt_embedding)

class UniRISKModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.K = config['risk']['K']
        
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            config['model']['policy_model'],
            torch_dtype=torch.float32,
            device_map="auto"
        )
        
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            config['model']['reward_model'],
            num_labels=self.K
        )
        
        hidden_size = self.policy_model.config.hidden_size
        self.viqn = VIQN(hidden_size, self.K)
        self.meta_risk_adapter = MetaRiskAdapter(hidden_size, self.K)
        
    def forward(self, input_ids, attention_mask, taus):
        outputs = self.policy_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        pooled = hidden_states.mean(dim=1)
        
        q_values = self.viqn(pooled, taus)
        risk_weights = self.meta_risk_adapter(pooled)
        
        return q_values, risk_weights, outputs.logits

def pareto_gap(qw, ql, psi):
    diff = qw - ql.flip(1)
    return (psi * diff).sum((1, 2))

def compute_conformal_delta(residuals, epsilon=0.05):
    return torch.quantile(residuals.abs(), 1 - epsilon, dim=0)

class UniRISKLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.beta = config['risk']['beta']
        self.alpha = config['risk']['alpha']
        self.gamma = config['risk']['gamma']
        
    def forward(self, q_values, risk_weights, logits, targets, true_rewards=None):
        batch_size, n_tau, K = q_values.shape
        
        qw = q_values[:, :n_tau//2, :]
        ql = q_values[:, n_tau//2:, :]
        
        psi = risk_weights.unsqueeze(1).expand(-1, n_tau//2, -1)
        spectral_risk = pareto_gap(qw, ql, psi).mean()
        
        policy_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        cert_loss = 0.0
        if true_rewards is not None:
            cvar_est = q_values[q_values >= 0.9].mean(dim=1)
            residuals = true_rewards - cvar_est
            delta = compute_conformal_delta(residuals)
            budget = torch.tensor([0.05, 0.05, 0.05, 0.05]).to(q_values.device)
            cert_loss = torch.clamp(cvar_est.mean(0) + delta - budget, min=0).sum()
        
        total_loss = policy_loss + self.beta * spectral_risk + self.gamma * cert_loss
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'spectral_risk': spectral_risk,
            'cert_loss': cert_loss
        }

class UniRISKTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = UniRISKModel(config).to(self.device)
        self.loss_fn = UniRISKLoss(config)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config['experiment']['learning_rate']),
            weight_decay=0.1
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['policy_model'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def train(self, train_dataset, val_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.config['experiment']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['experiment']['batch_size'])
        
        best_val_score = float('-inf')
        training_losses = []
        
        for epoch in range(self.config['experiment']['epochs']):
            self.model.train()
            epoch_losses = []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['experiment']['epochs']}")
            for batch_idx, batch in enumerate(pbar):
                try:
                    self.optimizer.zero_grad()
                    
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    max_len = min(input_ids.size(1), self.config['data']['max_length'])
                    input_ids = input_ids[:, :max_len]
                    attention_mask = attention_mask[:, :max_len]
                    
                    n_tau = self.config['risk']['tau_samples']
                    taus = torch.rand(input_ids.size(0), n_tau, 1).to(self.device)
                    
                    print(f"Processing batch {batch_idx}, input shape: {input_ids.shape}")
                    
                    q_values, risk_weights, logits = self.model(input_ids, attention_mask, taus)
                    loss_dict = self.loss_fn(q_values, risk_weights, logits, input_ids)
                    loss = loss_dict['total_loss']
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    
                    epoch_losses.append(loss.item())
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                    
                    if batch_idx % self.config['output']['log_interval'] == 0:
                        print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
                    
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
            
            avg_loss = np.mean(epoch_losses)
            training_losses.append(avg_loss)
            
            val_score = self.validate(val_loader)
            print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Val Score = {val_score:.4f}")
            
            if val_score > best_val_score:
                best_val_score = val_score
                torch.save(self.model.state_dict(), f"{self.config['output']['save_dir']}/best_model.pt")
        
        self.plot_training_curves(training_losses)
        
        return self.model, {
            'epochs': self.config['experiment']['epochs'],
            'final_loss': training_losses[-1],
            'best_val_score': best_val_score,
            'training_losses': training_losses
        }
    
    def validate(self, val_loader):
        self.model.eval()
        total_score = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                n_tau = self.config['risk']['tau_samples']
                taus = torch.rand(input_ids.size(0), n_tau, 1).to(self.device)
                
                q_values, risk_weights, logits = self.model(input_ids, attention_mask, taus)
                
                score = -q_values.mean().item()
                total_score += score
                num_batches += 1
        
        return total_score / num_batches if num_batches > 0 else 0.0
    
    def plot_training_curves(self, losses):
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('UniRISK-T Training Curves')
        plt.legend()
        plt.grid(True)
        
        save_path = f"{self.config['output']['save_dir']}/images/training_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {save_path}")
        return save_path
