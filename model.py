import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import math


# Load and preprocess data
def load_and_preprocess_data(file_path):
    """Load and preprocess the retail inventory data"""
    df = pd.read_csv(
        r"retail_store_inventory.csv"
    )

    # Convert date to datetime and extract features
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    # Encode categorical variables
    le_store = LabelEncoder()
    le_product = LabelEncoder()
    le_category = LabelEncoder()
    le_region = LabelEncoder()
    le_weather = LabelEncoder()
    le_seasonality = LabelEncoder()

    df["Store_ID_encoded"] = le_store.fit_transform(df["Store ID"])
    df["Product_ID_encoded"] = le_product.fit_transform(df["Product ID"])
    df["Category_encoded"] = le_category.fit_transform(df["Category"])
    df["Region_encoded"] = le_region.fit_transform(df["Region"])
    df["Weather_encoded"] = le_weather.fit_transform(df["Weather Condition"])
    df["Seasonality_encoded"] = le_seasonality.fit_transform(df["Seasonality"])

    return df


# Define GRU Model
class GRUDemandForecastModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(GRUDemandForecastModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU layers
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Reshape input for GRU (batch_size, seq_length=1, input_size)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward pass through GRU
        out, _ = self.gru(x, h0)

        # Take the last output
        out = out[:, -1, :]

        # Forward pass through fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


# Define LSTM Model
class LSTMDemandForecastModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(LSTMDemandForecastModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Reshape input for LSTM (batch_size, seq_length=1, input_size)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Take the last output
        out = out[:, -1, :]

        # Forward pass through fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


# Define Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


# Define Transformer Model
class TransformerDemandForecastModel(nn.Module):
    def __init__(
        self, input_size, d_model=128, nhead=8, num_layers=3, num_classes=2, dropout=0.2
    ):
        super(TransformerDemandForecastModel, self).__init__()
        self.d_model = d_model
        self.input_size = input_size

        # Input projection layer
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        # Reshape input if needed (batch_size, seq_length, input_size)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        batch_size, seq_len, _ = x.shape

        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)

        # Add positional encoding
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, d_model)

        # Apply transformer encoder
        transformer_output = self.transformer_encoder(
            x
        )  # (batch_size, seq_len, d_model)

        # Global average pooling or take the last token
        pooled_output = transformer_output.mean(dim=1)  # (batch_size, d_model)

        # Classification
        output = self.classifier(pooled_output)

        return output


# Training function
def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    num_epochs=50,
    device="cpu",
    model_name="Model",
):
    """Train the model"""
    model.train()
    train_losses = []

    print(f"Training {model_name}...")

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"{model_name} - Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}"
            )

    return train_losses


# Evaluation function
def evaluate_model(model, test_loader, device="cpu"):
    """Evaluate the model and return predictions"""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    return all_targets, all_predictions


# Function to calculate ROC AUC and confusion matrix
def get_roc_auc_confusion_matrix(model, dataloader, device="cpu"):
    """Calculate ROC AUC and confusion matrix for a single model"""
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            probs = nn.functional.softmax(outputs, dim=1)[
                :, 1
            ]  # Probability of class 1
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    roc_auc = roc_auc_score(all_targets, all_probs)

    # Get predicted class from model output
    preds = (np.array(all_probs) >= 0.5).astype(int)
    cm = confusion_matrix(all_targets, preds)

    return roc_auc, cm, all_targets, all_probs


# Function to combine predictions of three models using soft voting
def combine_models_predictions(models, dataloader, device="cpu"):
    """Combine predictions from multiple models using ensemble voting"""
    for model in models:
        model.eval()

    all_targets = []
    combined_probs = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            # Collect probabilities from all models
            probs_models = []
            for model in models:
                outputs = model(data)
                probs = nn.functional.softmax(outputs, dim=1)[
                    :, 1
                ]  # Positive class probabilities
                probs_models.append(probs.cpu().numpy())

            # Average the probabilities (soft voting)
            avg_probs = np.mean(probs_models, axis=0)
            combined_probs.extend(avg_probs)
            all_targets.extend(target.cpu().numpy())

    roc_auc = roc_auc_score(all_targets, combined_probs)
    preds = (np.array(combined_probs) >= 0.5).astype(int)
    cm = confusion_matrix(all_targets, preds)

    return roc_auc, cm, all_targets, combined_probs


# Function to plot confusion matrix
def plot_confusion_matrix(cm, title, class_names=["Low Demand", "High Demand"]):
    """Plot confusion matrix"""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# Calculate metrics
def calculate_metrics(y_true, y_pred):
    """Calculate all evaluation metrics"""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1_Score": f1_score(y_true, y_pred, average="weighted"),
    }
    return metrics


# Enhanced model comparison function
def compare_all_models(model_metrics_dict, model_names):
    """Compare all models including ensemble"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL COMPARISON RESULTS")
    print("=" * 80)

    comparison_df = pd.DataFrame(model_metrics_dict).T
    comparison_df = comparison_df.round(4)
    print(comparison_df)

    # Find best model for each metric
    print("\n" + "-" * 80)
    print("BEST PERFORMING MODEL FOR EACH METRIC:")
    print("-" * 80)

    for metric in comparison_df.columns:
        best_model = comparison_df[metric].idxmax()
        best_score = comparison_df[metric].max()
        print(f"{metric:12}: {best_model:15} (Score: {best_score:.4f})")

    return comparison_df


# Main implementation
def main():
    # Load and preprocess data
    df = load_and_preprocess_data("retail_store_inventory.csv")

    # Select features for the model
    feature_columns = [
        "Store_ID_encoded",
        "Product_ID_encoded",
        "Category_encoded",
        "Region_encoded",
        "Day",
        "Month",
        "Year",
        "DayOfWeek",
        "Inventory Level",
        "Units Sold",
        "Units Ordered",
        "Price",
        "Discount",
        "Competitor Pricing",
        "Weather_encoded",
        "Holiday/Promotion",
        "Seasonality_encoded",
    ]

    X = df[feature_columns].values

    # Create binary classification target (high/low demand)
    y_continuous = df["Demand Forecast"].values
    median_demand = np.median(y_continuous)
    y = (y_continuous >= median_demand).astype(int)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model parameters
    input_size = X_train_scaled.shape[1]
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize all models
    gru_model = GRUDemandForecastModel(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3,
    ).to(device)

    lstm_model = LSTMDemandForecastModel(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3,
    ).to(device)

    transformer_model = TransformerDemandForecastModel(
        input_size=input_size,
        d_model=128,
        nhead=8,
        num_layers=3,
        num_classes=num_classes,
        dropout=0.3,
    ).to(device)

    # Loss function and optimizers
    criterion = nn.CrossEntropyLoss()
    gru_optimizer = torch.optim.Adam(
        gru_model.parameters(), lr=0.001, weight_decay=1e-5
    )
    lstm_optimizer = torch.optim.Adam(
        lstm_model.parameters(), lr=0.001, weight_decay=1e-5
    )
    transformer_optimizer = torch.optim.Adam(
        transformer_model.parameters(), lr=0.001, weight_decay=1e-5
    )

    num_epochs = 50

    # Train all models
    print("=" * 60)
    print("TRAINING ALL MODELS")
    print("=" * 60)

    gru_losses = train_model(
        gru_model,
        train_loader,
        criterion,
        gru_optimizer,
        num_epochs=num_epochs,
        device=device,
        model_name="GRU",
    )

    lstm_losses = train_model(
        lstm_model,
        train_loader,
        criterion,
        lstm_optimizer,
        num_epochs=num_epochs,
        device=device,
        model_name="LSTM",
    )

    transformer_losses = train_model(
        transformer_model,
        train_loader,
        criterion,
        transformer_optimizer,
        num_epochs=num_epochs,
        device=device,
        model_name="Transformer",
    )

    # Evaluate all individual models
    print("\n" + "=" * 60)
    print("EVALUATING INDIVIDUAL MODELS")
    print("=" * 60)

    # GRU evaluation
    print("\nEvaluating GRU Model...")
    gru_y_true, gru_y_pred = evaluate_model(gru_model, test_loader, device)
    gru_metrics = calculate_metrics(gru_y_true, gru_y_pred)
    gru_roc_auc, gru_cm, _, _ = get_roc_auc_confusion_matrix(
        gru_model, test_loader, device
    )
    gru_metrics["ROC_AUC"] = gru_roc_auc

    # LSTM evaluation
    print("Evaluating LSTM Model...")
    lstm_y_true, lstm_y_pred = evaluate_model(lstm_model, test_loader, device)
    lstm_metrics = calculate_metrics(lstm_y_true, lstm_y_pred)
    lstm_roc_auc, lstm_cm, _, _ = get_roc_auc_confusion_matrix(
        lstm_model, test_loader, device
    )
    lstm_metrics["ROC_AUC"] = lstm_roc_auc

    # Transformer evaluation
    print("Evaluating Transformer Model...")
    transformer_y_true, transformer_y_pred = evaluate_model(
        transformer_model, test_loader, device
    )
    transformer_metrics = calculate_metrics(transformer_y_true, transformer_y_pred)
    transformer_roc_auc, transformer_cm, _, _ = get_roc_auc_confusion_matrix(
        transformer_model, test_loader, device
    )
    transformer_metrics["ROC_AUC"] = transformer_roc_auc

    # Ensemble model evaluation
    print("Evaluating Ensemble Model (Combined)...")
    models = [gru_model, lstm_model, transformer_model]
    ensemble_roc_auc, ensemble_cm, ensemble_targets, ensemble_probs = (
        combine_models_predictions(models, test_loader, device)
    )
    ensemble_preds = (np.array(ensemble_probs) >= 0.5).astype(int)
    ensemble_metrics = calculate_metrics(ensemble_targets, ensemble_preds)
    ensemble_metrics["ROC_AUC"] = ensemble_roc_auc

    # Print individual model results
    print("\n" + "=" * 60)
    print("INDIVIDUAL MODEL RESULTS")
    print("=" * 60)

    # GRU Results
    print("\nGRU MODEL RESULTS:")
    print("-" * 30)
    print(f"Accuracy:  {gru_metrics['Accuracy']:.4f}")
    print(f"Precision: {gru_metrics['Precision']:.4f}")
    print(f"Recall:    {gru_metrics['Recall']:.4f}")
    print(f"F1 Score:  {gru_metrics['F1_Score']:.4f}")
    print(f"ROC AUC:   {gru_metrics['ROC_AUC']:.4f}")

    # LSTM Results
    print("\nLSTM MODEL RESULTS:")
    print("-" * 30)
    print(f"Accuracy:  {lstm_metrics['Accuracy']:.4f}")
    print(f"Precision: {lstm_metrics['Precision']:.4f}")
    print(f"Recall:    {lstm_metrics['Recall']:.4f}")
    print(f"F1 Score:  {lstm_metrics['F1_Score']:.4f}")
    print(f"ROC AUC:   {lstm_metrics['ROC_AUC']:.4f}")

    # Transformer Results
    print("\nTRANSFORMER MODEL RESULTS:")
    print("-" * 30)
    print(f"Accuracy:  {transformer_metrics['Accuracy']:.4f}")
    print(f"Precision: {transformer_metrics['Precision']:.4f}")
    print(f"Recall:    {transformer_metrics['Recall']:.4f}")
    print(f"F1 Score:  {transformer_metrics['F1_Score']:.4f}")
    print(f"ROC AUC:   {transformer_metrics['ROC_AUC']:.4f}")

    # Ensemble Results
    print("\nENSEMBLE MODEL RESULTS:")
    print("-" * 30)
    print(f"Accuracy:  {ensemble_metrics['Accuracy']:.4f}")
    print(f"Precision: {ensemble_metrics['Precision']:.4f}")
    print(f"Recall:    {ensemble_metrics['Recall']:.4f}")
    print(f"F1 Score:  {ensemble_metrics['F1_Score']:.4f}")
    print(f"ROC AUC:   {ensemble_metrics['ROC_AUC']:.4f}")

    # Compare all models
    all_metrics = {
        "GRU": gru_metrics,
        "LSTM": lstm_metrics,
        "Transformer": transformer_metrics,
        "Ensemble": ensemble_metrics,
    }

    comparison_df = compare_all_models(
        all_metrics, ["GRU", "LSTM", "Transformer", "Ensemble"]
    )

    # Plot confusion matrices
    print("\n" + "=" * 60)
    print("CONFUSION MATRICES")
    print("=" * 60)

    plot_confusion_matrix(gru_cm, "GRU Model - Confusion Matrix")
    plot_confusion_matrix(lstm_cm, "LSTM Model - Confusion Matrix")
    plot_confusion_matrix(transformer_cm, "Transformer Model - Confusion Matrix")
    plot_confusion_matrix(ensemble_cm, "Ensemble Model - Confusion Matrix")

    # Plot comprehensive comparison
    plt.figure(figsize=(15, 10))

    # Training losses comparison
    plt.subplot(2, 2, 1)
    plt.plot(gru_losses, label="GRU", color="blue", linewidth=2)
    plt.plot(lstm_losses, label="LSTM", color="green", linewidth=2)
    plt.plot(transformer_losses, label="Transformer", color="red", linewidth=2)
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Metrics comparison
    plt.subplot(2, 2, 2)
    metrics_names = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]

    gru_values = [gru_metrics[m] for m in metrics_names]
    lstm_values = [lstm_metrics[m] for m in metrics_names]
    transformer_values = [transformer_metrics[m] for m in metrics_names]
    ensemble_values = [ensemble_metrics[m] for m in metrics_names]

    x = np.arange(len(metrics_names))
    width = 0.2

    plt.bar(x - 1.5 * width, gru_values, width, label="GRU", color="blue", alpha=0.7)
    plt.bar(x - 0.5 * width, lstm_values, width, label="LSTM", color="green", alpha=0.7)
    plt.bar(
        x + 0.5 * width,
        transformer_values,
        width,
        label="Transformer",
        color="red",
        alpha=0.7,
    )
    plt.bar(
        x + 1.5 * width,
        ensemble_values,
        width,
        label="Ensemble",
        color="purple",
        alpha=0.7,
    )

    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.xticks(x, metrics_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    # ROC AUC comparison (separate plot)
    plt.subplot(2, 2, 3)
    model_names = ["GRU", "LSTM", "Transformer", "Ensemble"]
    roc_scores = [gru_roc_auc, lstm_roc_auc, transformer_roc_auc, ensemble_roc_auc]
    colors = ["blue", "green", "red", "purple"]

    bars = plt.bar(model_names, roc_scores, color=colors, alpha=0.7)
    plt.title("ROC AUC Score Comparison")
    plt.ylabel("ROC AUC Score")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, score in zip(bars, roc_scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.show()

    # Print detailed classification reports
    print("\n" + "=" * 60)
    print("DETAILED CLASSIFICATION REPORTS")
    print("=" * 60)

    print("\nGRU Classification Report:")
    print(
        classification_report(
            gru_y_true, gru_y_pred, target_names=["Low Demand", "High Demand"]
        )
    )

    print("\nLSTM Classification Report:")
    print(
        classification_report(
            lstm_y_true, lstm_y_pred, target_names=["Low Demand", "High Demand"]
        )
    )

    print("\nTransformer Classification Report:")
    print(
        classification_report(
            transformer_y_true,
            transformer_y_pred,
            target_names=["Low Demand", "High Demand"],
        )
    )

    print("\nEnsemble Classification Report:")
    print(
        classification_report(
            ensemble_targets, ensemble_preds, target_names=["Low Demand", "High Demand"]
        )
    )

    return {
        "gru_model": gru_model,
        "lstm_model": lstm_model,
        "transformer_model": transformer_model,
        "gru_metrics": gru_metrics,
        "lstm_metrics": lstm_metrics,
        "transformer_metrics": transformer_metrics,
        "ensemble_metrics": ensemble_metrics,
        "gru_losses": gru_losses,
        "lstm_losses": lstm_losses,
        "transformer_losses": transformer_losses,
        "comparison_df": comparison_df,
        "confusion_matrices": {
            "gru": gru_cm,
            "lstm": lstm_cm,
            "transformer": transformer_cm,
            "ensemble": ensemble_cm,
        },
    }


if __name__ == "__main__":
    results = main()
