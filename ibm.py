# -*- coding: utf-8 -*-
"""Customer churn model training and evaluation."""

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from torch.utils.data import Dataset, DataLoader

print("Libraries imported!")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

root = tk.Tk()
root.withdraw()
csv_path = filedialog.askopenfilename(
    title="Select the Telco Customer Churn CSV file",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)
root.destroy()

if not csv_path:
    raise FileNotFoundError("No CSV file was selected.")

df = pd.read_csv(csv_path)

print("Dataset shape:", df.shape)

df.head()

df.info()

print(df.isnull().sum())

print(df["Churn"].value_counts())

sns.countplot(data=df, x="Churn")

plt.title("Customer Churn Distribution")
plt.show()

# Remove customer ID
df = df.drop(columns=["customerID"])

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Handle missing values
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Convert target into binary
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

X = df.drop(columns=["Churn"])
y = df["Churn"]

print("Features:", X.shape)
print("Target:", y.shape)

print(df.head())

print(df.isnull().sum().sum())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training:", X_train.shape)
print("Testing:", X_test.shape)

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("Categorical Features:")
print(categorical_features)

print("\nNumerical Features:")
print(numerical_features)

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

X_train_cat = encoder.fit_transform(
    X_train[categorical_features].fillna("Unknown")
)

X_test_cat = encoder.transform(
    X_test[categorical_features].fillna("Unknown")
)

# Shift IDs by 1 so unknown values become 0
X_train_cat = X_train_cat.astype(int) + 1
X_test_cat = X_test_cat.astype(int) + 1

print(X_train_cat[:3])

category_sizes = [
    len(categories) + 1
    for categories in encoder.categories_
]

print("Category sizes:", category_sizes)

scaler = StandardScaler()

X_train_num = scaler.fit_transform(
    X_train[numerical_features]
)

X_test_num = scaler.transform(
    X_test[numerical_features]
)

print("Numerical scaling completed!")

X_train_cat = torch.tensor(
    X_train_cat,
    dtype=torch.long
)

X_test_cat = torch.tensor(
    X_test_cat,
    dtype=torch.long
)

X_train_num = torch.tensor(
    X_train_num,
    dtype=torch.float32
)

X_test_num = torch.tensor(
    X_test_num,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train.values,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test.values,
    dtype=torch.float32
)

class ChurnDataset(Dataset):

    def __init__(self, cat, num, target):

        self.cat = cat
        self.num = num
        self.target = target

    def __len__(self):

        return len(self.target)

    def __getitem__(self, index):

        return (
            self.cat[index],
            self.num[index],
            self.target[index]
        )


train_dataset = ChurnDataset(
    X_train_cat,
    X_train_num,
    y_train_tensor
)

test_dataset = ChurnDataset(
    X_test_cat,
    X_test_num,
    y_test_tensor
)


train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

print("DataLoaders created!")

class TabTransformer(nn.Module):

    def __init__(
        self,
        category_sizes,
        num_numeric,
        embedding_dim=32,
        num_heads=4,
        num_layers=2
    ):

        super().__init__()

        # Create embedding for every categorical feature
        self.embeddings = nn.ModuleList([

            nn.Embedding(size, embedding_dim)

            for size in category_sizes

        ])

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(

            nn.Linear(
                embedding_dim + num_numeric,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Linear(32, 1)

        )


    def forward(self, x_cat, x_num):

        # Generate categorical embeddings
        embeddings = [

            embedding(x_cat[:, i])

            for i, embedding in enumerate(self.embeddings)

        ]

        # Combine categorical embeddings
        x = torch.stack(
            embeddings,
            dim=1
        )

        # Transformer
        x = self.transformer(x)

        # Mean pooling
        x = x.mean(dim=1)

        # Combine categorical and numerical features
        x = torch.cat(
            [x, x_num],
            dim=1
        )

        # Prediction
        output = self.classifier(x)

        return output.squeeze(1)

model = TabTransformer(
    category_sizes=category_sizes,
    num_numeric=len(numerical_features),
    embedding_dim=32,
    num_heads=4,
    num_layers=2
).to(device)

print(model)

positive_weight = torch.tensor(
    [(y_train == 0).sum() / (y_train == 1).sum()],
    dtype=torch.float32
).to(device)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=positive_weight
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

EPOCHS = 30

train_losses = []

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for cat_batch, num_batch, y_batch in train_loader:

        cat_batch = cat_batch.to(device)
        num_batch = num_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward propagation
        outputs = model(
            cat_batch,
            num_batch
        )

        # Calculate loss
        loss = criterion(
            outputs,
            y_batch
        )

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    train_losses.append(avg_loss)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {avg_loss:.4f}"
    )

plt.plot(train_losses)

plt.title("TabTransformer Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid()

plt.show()

model.eval()

all_predictions = []
all_probabilities = []
all_actuals = []

with torch.no_grad():

    for cat_batch, num_batch, y_batch in test_loader:

        cat_batch = cat_batch.to(device)
        num_batch = num_batch.to(device)

        outputs = model(
            cat_batch,
            num_batch
        )

        probabilities = torch.sigmoid(outputs)

        predictions = (
            probabilities >= 0.5
        ).int()

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_actuals.extend(
            y_batch.numpy()
        )


all_predictions = np.array(all_predictions)
all_probabilities = np.array(all_probabilities)
all_actuals = np.array(all_actuals)

print("Accuracy:",
      accuracy_score(all_actuals, all_predictions))

print("Precision:",
      precision_score(all_actuals, all_predictions, zero_division=0))

print("Recall:",
      recall_score(all_actuals, all_predictions, zero_division=0))

print("F1 Score:",
      f1_score(all_actuals, all_predictions, zero_division=0))

print("ROC-AUC:",
      roc_auc_score(all_actuals, all_probabilities))

print(
    classification_report(
        all_actuals,
        all_predictions,
        target_names=["No Churn", "Churn"]
    )
)

cm = confusion_matrix(
    all_actuals,
    all_predictions
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title("Customer Churn Confusion Matrix")

plt.show()

new_customer = pd.DataFrame([{

    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.50,
    "TotalCharges": 286.50

}])

new_cat = encoder.transform(
    new_customer[categorical_features].fillna("Unknown")
)

new_cat = torch.tensor(
    new_cat.astype(int) + 1,
    dtype=torch.long
).to(device)


new_num = scaler.transform(
    new_customer[numerical_features]
)

new_num = torch.tensor(
    new_num,
    dtype=torch.float32
).to(device)

model.eval()

with torch.no_grad():

    output = model(
        new_cat,
        new_num
    )

    probability = torch.sigmoid(output).item()


print("Customer Churn Probability:",
      round(probability * 100, 2), "%")

if probability >= 0.5:

    print("Prediction: Customer likely to churn")

else:

    print("Prediction: Customer likely to stay")

import joblib

torch.save(
    model.state_dict(),
    "tabtransformer_churn.pth"
)

joblib.dump(
    scaler,
    "numerical_scaler.pkl"
)

joblib.dump(
    encoder,
    "categorical_encoder.pkl"
)

print("All files saved!")

from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(
    np.concatenate([
        X_train_num.numpy(),
        X_train_cat.numpy()
    ], axis=1),
    y_train.to_numpy()
)

print("XGBoost training completed!")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Ensure XGBoost is fitted with the correct binary target and combined features
X_train_combined = np.concatenate([
    X_train_num.numpy(),
    X_train_cat.numpy()
], axis=1)

# Use y_train_tensor (which is already binary float32) or y_train mapped values
xgb_model.fit(X_train_combined, y_train_tensor.numpy())

# XGBoost predictions
X_test_combined = np.concatenate([
    X_test_num.numpy(),
    X_test_cat.numpy()
], axis=1)

xgb_prob = xgb_model.predict_proba(
    X_test_combined
)[:, 1]

xgb_pred = (xgb_prob >= 0.5).astype(int)

# Transformer predictions
transformer_prob = all_probabilities
transformer_pred = (
    transformer_prob >= 0.5
).astype(int)

# Compare
results = pd.DataFrame({
    "Model": ["XGBoost", "Transformer"],
    "Accuracy": [
        accuracy_score(all_actuals, xgb_pred),
        accuracy_score(all_actuals, transformer_pred)
    ],
    "Precision": [
        precision_score(all_actuals, xgb_pred, zero_division=0),
        precision_score(all_actuals, transformer_pred, zero_division=0)
    ],
    "Recall": [
        recall_score(all_actuals, xgb_pred, zero_division=0),
        recall_score(all_actuals, transformer_pred, zero_division=0)
    ],
    "F1": [
        f1_score(all_actuals, xgb_pred, zero_division=0),
        f1_score(all_actuals, transformer_pred, zero_division=0)
    ],
    "ROC-AUC": [
        roc_auc_score(all_actuals, xgb_prob),
        roc_auc_score(all_actuals, transformer_prob)
    ]
})

print(results)
