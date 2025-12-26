import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
from config import *

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s,.\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_data(file_path):
    df = pd.read_csv(file_path)
    df['property_address'] = df['property_address'].apply(clean_text)
    df['label'] = df['categories'].map(LABEL2ID)
    return df['property_address'].tolist(), df['label'].tolist()


class PropertyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def train_model(model, train_loader, val_loader, device):
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    best_acc = 0
    print("TRAINING")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            progress.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_acc = correct / total
        avg_loss = train_loss / len(train_loader)
        
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        
        print(f"\nEpoch {epoch+1}: Train Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained(MODEL_DIR)
            print(f"Validation Accuracy: {val_acc:.4f}\n")
    
    return best_acc

def evaluate_model(model, data_loader, device):
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    report = classification_report(all_labels, all_preds, target_names=CATEGORIES, digits=4)
    accuracy = accuracy_score(all_labels, all_preds)
    
    print("\nCLASSIFICATION REPORT")
    print(report)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    with open(f'{RESULTS_DIR}/classification_report.txt', 'w') as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n")
        f.write(report)
        f.write(f"\n\nOverall Accuracy: {accuracy:.4f}\n")
    
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"\nResults saved to: {RESULTS_DIR}/")
    
    return accuracy

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    trust_remote_code=True
)
    
    train_texts, train_labels = load_data(f'{DATA_DIR}/train.csv')
    val_texts, val_labels = load_data(f'{DATA_DIR}/val.csv')
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    
    train_dataset = PropertyDataset(train_texts, train_labels, tokenizer)
    val_dataset = PropertyDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    print(f"\nLoading model: {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    ).to(DEVICE)
    
    best_acc = train_model(model, train_loader, val_loader, DEVICE)
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    
    final_acc = evaluate_model(model, val_loader, DEVICE)
    
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print(f"Final Accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    main()