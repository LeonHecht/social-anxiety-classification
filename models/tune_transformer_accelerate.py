import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

class WeightedSmoothCrossEntropyLoss(torch.nn.Module):
    def __init__(self, class_weights, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.class_weights = torch.tensor(class_weights, dtype=torch.float)  # Ensure it's on the correct device in forward.

    def forward(self, inputs, targets):
        n_classes = inputs.size(1)
        targets = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.smoothing) * targets + self.smoothing / n_classes
        if self.class_weights is not None:
            targets *= self.class_weights.unsqueeze(0)
        loss = torch.nn.functional.log_softmax(inputs, dim=1)  # Compute log softmax
        loss = -(targets * loss).sum(dim=1).mean()  # Compute cross-entropy
        return loss

class WeightedAutoModel(AutoModelForSequenceClassification):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float, device=self.device)
            self.loss_fct = WeightedSmoothCrossEntropyLoss(class_weights=self.class_weights)
        else:
            self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids, attention_mask=attention_mask, **kwargs)
        logits = outputs.logits

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs.loss = loss
        return outputs

def load_model(model_checkpoint, num_labels, classes, y_train):
    from transformers import AutoConfig
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    # Load configuration and update it with the number of labels
    config = AutoConfig.from_pretrained(model_checkpoint, num_labels=num_labels)
    # Load model with the updated configuration
    model = WeightedAutoModel.from_pretrained(model_checkpoint, config=config)
    # If you have custom class weights handling inside the model, you might need to set it after instantiation
    model.class_weights = torch.tensor(class_weights, dtype=torch.float, device=model.device)
    model.loss_fct = WeightedSmoothCrossEntropyLoss(class_weights=model.class_weights)
    return model

def compute_metrics(logits, labels):
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def tokenize_data(tokenizer, texts, labels=None):
    """ Tokenizes texts and converts the data to tensors, including labels if provided. """
    # Tokenize text
    encoding = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    
    if labels is not None:
        # Ensure labels are provided as a list, then convert them to tensors
        labels = labels.to_list() if hasattr(labels, 'to_list') else labels
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        encoding['labels'] = labels_tensor

    return encoding

def run(model_checkpoint, num_labels, train_texts, val_texts, test_texts, y_train, y_val, y_test):
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = load_model(model_checkpoint, num_labels, np.unique(y_train), y_train)
    
    train_dataset = tokenize_data(tokenizer, train_texts.to_list(), y_train.to_list())
    val_dataset = tokenize_data(tokenizer, val_texts.to_list(), y_val.to_list())
    test_dataset = tokenize_data(tokenizer, test_texts.to_list(), y_test.to_list())

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    model, train_loader, val_loader, test_loader = accelerator.prepare(model, train_loader, val_loader, test_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.5e-6)

    best_f1 = 0
    best_model = None

    for epoch in range(10):
        model.train()
        for batch in train_loader:
            outputs = model(**batch)
            loss = outputs['loss']
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        
        model.eval()
        with torch.no_grad():
            all_logits, all_labels = [], []
            for batch in val_loader:
                outputs = model(**batch)
                all_logits.append(outputs['logits'])
                all_labels.append(batch['labels'])
            
            all_logits = torch.cat(all_logits)
            all_labels = torch.cat(all_labels)
            metrics = compute_metrics(all_logits.cpu().numpy(), all_labels.cpu().numpy())
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_model = model.state_dict()

        print(f"Epoch {epoch+1} - Val F1: {metrics['f1']:.4f}")

    if best_model is not None:
        model.load_state_dict(best_model)
        model.eval()
        all_logits = []
        for batch in test_loader:
            with torch.no_grad():
                outputs = model(**batch)
            all_logits.append(outputs['logits'])
        
        all_logits = torch.cat(all_logits)
        predictions = np.argmax(all_logits.cpu().numpy(), axis=1)
        return predictions

    return None