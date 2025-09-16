import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer
import numpy as np
from datasets import load_dataset
import random

class SafetyDataset(Dataset):
    def __init__(self, texts, rewards, tokenizer, max_length=512):
        self.texts = texts
        self.rewards = rewards
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        reward = self.rewards[idx] if self.rewards is not None else [0.5, 0.5, 0.5, 0.5]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'rewards': torch.tensor(reward, dtype=torch.float32)
        }

def create_synthetic_safety_data(num_samples=1000):
    prompts = [
        "How can I help you today?",
        "What's the weather like?",
        "Tell me about artificial intelligence.",
        "How do I cook pasta?",
        "What are the benefits of exercise?",
        "Explain quantum physics simply.",
        "How to write a good essay?",
        "What is machine learning?",
        "How to stay healthy?",
        "Tell me a joke."
    ]
    
    responses = [
        "I'm here to assist you with any questions or tasks you might have.",
        "I don't have access to real-time weather data, but you can check a weather app.",
        "AI is a field of computer science focused on creating intelligent machines.",
        "Boil water, add salt, then add pasta and cook according to package directions.",
        "Exercise improves cardiovascular health, strength, and mental well-being.",
        "Quantum physics studies the behavior of matter and energy at the smallest scales.",
        "A good essay has a clear thesis, supporting evidence, and logical structure.",
        "Machine learning is a subset of AI that learns patterns from data.",
        "Maintain a balanced diet, exercise regularly, and get adequate sleep.",
        "Why don't scientists trust atoms? Because they make up everything!"
    ]
    
    texts = []
    rewards = []
    
    for i in range(num_samples):
        prompt = random.choice(prompts)
        response = random.choice(responses)
        text = f"Human: {prompt}\nAssistant: {response}"
        texts.append(text)
        
        toxicity = np.random.beta(2, 8)
        privacy = np.random.beta(2, 8) 
        jailbreak = np.random.beta(2, 8)
        bias = np.random.beta(2, 8)
        
        rewards.append([toxicity, privacy, jailbreak, bias])
    
    return texts, rewards

def load_ultrafeedback_data(max_samples=None):
    try:
        dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
        
        texts = []
        rewards = []
        
        for i, example in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
                
            prompt = example.get('prompt', '')
            chosen = example.get('chosen', [])
            
            if chosen and len(chosen) > 0:
                response = chosen[0].get('content', '') if isinstance(chosen[0], dict) else str(chosen[0])
                text = f"Human: {prompt}\nAssistant: {response}"
                texts.append(text)
                
                helpfulness = np.random.beta(6, 2)
                toxicity = np.random.beta(2, 8)
                privacy = np.random.beta(2, 8)
                jailbreak = np.random.beta(2, 8)
                
                rewards.append([helpfulness, toxicity, privacy, jailbreak])
        
        return texts, rewards
        
    except Exception as e:
        print(f"Failed to load UltraFeedback dataset: {e}")
        print("Using synthetic data instead...")
        return create_synthetic_safety_data(max_samples or 1000)

def load_mt_bench_prompts():
    try:
        dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        prompts = [example['prompt'] for example in dataset]
        return prompts[:100]
    except Exception as e:
        print(f"Failed to load MT-Bench prompts: {e}")
        return [
            "Write a short story about a robot learning to paint.",
            "Explain the concept of democracy to a 10-year-old.",
            "How would you solve climate change?",
            "What are the pros and cons of social media?",
            "Describe your ideal vacation destination."
        ]

def prepare_datasets(config):
    tokenizer = AutoTokenizer.from_pretrained(config['model']['policy_model'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    max_samples = config['data'].get('train_samples', 1000) + config['data'].get('val_samples', 200)
    
    print("Loading training data...")
    texts, rewards = load_ultrafeedback_data(max_samples)
    
    print(f"Loaded {len(texts)} text-reward pairs")
    
    dataset = SafetyDataset(texts, rewards, tokenizer, config['data']['max_length'])
    
    train_size = config['data']['train_samples']
    val_size = config['data']['val_samples']
    calib_size = min(500, len(dataset) - train_size - val_size)
    
    if train_size + val_size + calib_size > len(dataset):
        train_size = int(0.7 * len(dataset))
        val_size = int(0.2 * len(dataset))
        calib_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, calib_dataset = random_split(
        dataset, 
        [train_size, val_size, calib_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Calib: {len(calib_dataset)}")
    
    return train_dataset, val_dataset, calib_dataset
