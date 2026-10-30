import base64
from io import BytesIO

import matplotlib
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBClassifier

app = FastAPI(
    title="Customer Churn CSV API",
    description="Upload and inspect a customer churn CSV dataset.",
    version="1.0.0",
)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
PREVIEW_ROWS = 5
EPOCHS = 30
app.state.dataset = None
app.state.filename = None


class ChurnDataset(Dataset):
    def __init__(
        self,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        self.categorical = categorical
        self.numeric = numeric
        self.target = target

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.categorical[index], self.numeric[index], self.target[index]


class TabTransformer(nn.Module):
    def __init__(
        self,
        category_sizes: list[int],
        num_numeric: int,
        embedding_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(size, embedding_dim) for size in category_sizes]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim + num_numeric, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        embedded_features = [
            embedding(x_cat[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        x = torch.stack(embedded_features, dim=1)
        x = self.transformer(x).mean(dim=1)
        x = torch.cat([x, x_num], dim=1)
        return self.classifier(x).squeeze(1)


def _dataset_summary(dataset: pd.DataFrame, filename: str) -> dict:
    preview = (
        dataset.head(PREVIEW_ROWS)
        .astype(object)
        .where(pd.notna(dataset.head(PREVIEW_ROWS)), None)
        .to_dict(orient="records")
    )
    return {
        "filename": filename,
        "rows": len(dataset),
        "columns": dataset.columns.tolist(),
        "preview": jsonable_encoder(preview),
    }


def _target_as_binary(target: pd.Series) -> pd.Series:
    if target.isna().any():
        raise ValueError("The Churn column contains missing values.")

    if pd.api.types.is_numeric_dtype(target):
        binary_target = pd.to_numeric(target, errors="coerce")
    else:
        normalized = target.astype(str).str.strip().str.lower()
        binary_target = normalized.map(
            {"no": 0, "yes": 1, "false": 0, "true": 1, "0": 0, "1": 1}
        )
        if binary_target.isna().any():
            binary_target = pd.to_numeric(normalized, errors="coerce")

    if binary_target.isna().any() or set(binary_target.unique()) != {0, 1}:
        raise ValueError(
            "The Churn column must contain two classes represented by "
            "'No'/'Yes' or 0/1."
        )
    return binary_target.astype("int64")


def _metrics(actual: pd.Series, predicted, probabilities) -> dict:
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, probabilities)),
        "confusion_matrix": confusion_matrix(actual, predicted).tolist(),
        "classification_report": classification_report(
            actual,
            predicted,
            labels=[0, 1],
            target_names=["No Churn", "Churn"],
            output_dict=True,
            zero_division=0,
        ),
    }


def _chart_data(train_losses: list[float], transformer_metrics: dict) -> dict:
    matplotlib.use("Agg")
    charts = {}
    figure, axis = plt.subplots()
    axis.plot(range(1, len(train_losses) + 1), train_losses)
    axis.set(title="TabTransformer Training Loss", xlabel="Epoch", ylabel="Loss")
    axis.grid(True)
    loss_image = BytesIO()
    figure.savefig(loss_image, format="png", bbox_inches="tight")
    plt.close(figure)
    charts["training_loss_png_base64"] = base64.b64encode(
        loss_image.getvalue()
    ).decode("ascii")

    figure, axis = plt.subplots()
    image = axis.imshow(transformer_metrics["confusion_matrix"], cmap="Blues")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        xlabel="Predicted",
        ylabel="Actual",
        title="Transformer Confusion Matrix",
    )
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(transformer_metrics["confusion_matrix"][row][column]),
                ha="center",
                va="center",
                color="black",
            )
    figure.colorbar(image, ax=axis)
    matrix_image = BytesIO()
    figure.savefig(matrix_image, format="png", bbox_inches="tight")
    plt.close(figure)
    charts["confusion_matrix_png_base64"] = base64.b64encode(
        matrix_image.getvalue()
    ).decode("ascii")
    return charts


def _run_churn_model(dataset: pd.DataFrame) -> dict:
    if "Churn" not in dataset.columns:
        raise ValueError("The uploaded CSV must contain a 'Churn' target column.")

    data = dataset.copy()
    if "customerID" in data.columns:
        data = data.drop(columns=["customerID"])
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    target = _target_as_binary(data.pop("Churn"))
    if data.shape[1] == 0:
        raise ValueError("The CSV must include at least one feature besides Churn.")

    for column in data.select_dtypes(include=["number"]).columns:
        median = data[column].median()
        if pd.isna(median):
            raise ValueError(f"Feature '{column}' contains no usable numeric values.")
        data[column] = data[column].fillna(median)

    categorical_features = data.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numerical_features = data.select_dtypes(include=["number"]).columns.tolist()
    if not categorical_features:
        raise ValueError("The dataset must include at least one categorical feature.")
    data[categorical_features] = data[categorical_features].fillna("Unknown")

    class_counts = target.value_counts()
    if len(class_counts) != 2 or class_counts.min() < 2:
        raise ValueError("Each churn class needs at least two rows for evaluation.")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            data,
            target,
            test_size=0.2,
            random_state=42,
            stratify=target,
        )
    except ValueError as error:
        raise ValueError(f"Not enough rows for a stratified train/test split: {error}") from error

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    X_train_cat = encoder.fit_transform(X_train[categorical_features]).astype("int64") + 1
    X_test_cat = encoder.transform(X_test[categorical_features]).astype("int64") + 1
    category_sizes = [len(categories) + 1 for categories in encoder.categories_]

    if numerical_features:
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(X_train[numerical_features])
        X_test_num = scaler.transform(X_test[numerical_features])
    else:
        X_train_num = pd.DataFrame(index=X_train.index).to_numpy(dtype="float32")
        X_test_num = pd.DataFrame(index=X_test.index).to_numpy(dtype="float32")

    train_dataset = ChurnDataset(
        torch.tensor(X_train_cat, dtype=torch.long),
        torch.tensor(X_train_num, dtype=torch.float32),
        torch.tensor(y_train.to_numpy(), dtype=torch.float32),
    )
    test_dataset = ChurnDataset(
        torch.tensor(X_test_cat, dtype=torch.long),
        torch.tensor(X_test_num, dtype=torch.float32),
        torch.tensor(y_test.to_numpy(), dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = TabTransformer(
        category_sizes=category_sizes,
        num_numeric=len(numerical_features),
    ).to(device)
    positive_weight = torch.tensor(
        [(y_train == 0).sum() / (y_train == 1).sum()],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )

    train_losses = []
    for _ in range(EPOCHS):
        transformer.train()
        total_loss = 0.0
        for cat_batch, num_batch, y_batch in train_loader:
            cat_batch = cat_batch.to(device)
            num_batch = num_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = transformer(cat_batch, num_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

    transformer.eval()
    probabilities = []
    with torch.no_grad():
        for cat_batch, num_batch, _ in test_loader:
            outputs = transformer(cat_batch.to(device), num_batch.to(device))
            probabilities.extend(torch.sigmoid(outputs).cpu().numpy())

    transformer_probabilities = pd.Series(probabilities, index=y_test.index)
    transformer_predictions = (transformer_probabilities >= 0.5).astype("int64")
    transformer_metrics = _metrics(
        y_test,
        transformer_predictions,
        transformer_probabilities,
    )

    train_numeric = X_train_num
    test_numeric = X_test_num
    X_train_combined = pd.concat(
        [
            pd.DataFrame(train_numeric, index=X_train.index),
            pd.DataFrame(X_train_cat, index=X_train.index),
        ],
        axis=1,
    ).to_numpy()
    X_test_combined = pd.concat(
        [
            pd.DataFrame(test_numeric, index=X_test.index),
            pd.DataFrame(X_test_cat, index=X_test.index),
        ],
        axis=1,
    ).to_numpy()
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    xgb_model.fit(X_train_combined, y_train.to_numpy())
    xgb_probabilities = xgb_model.predict_proba(X_test_combined)[:, 1]
    xgb_predictions = (xgb_probabilities >= 0.5).astype("int64")
    xgb_metrics = _metrics(y_test, xgb_predictions, xgb_probabilities)

    return {
        "device": str(device),
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "epochs": EPOCHS,
        "training_loss": train_losses,
        "models": {
            "transformer": transformer_metrics,
            "xgboost": xgb_metrics,
        },
        "charts": _chart_data(train_losses, transformer_metrics),
    }


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"CSV file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")

    try:
        dataset = pd.read_csv(BytesIO(content))
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read the uploaded file as a CSV: {error}",
        ) from error

    if dataset.empty or len(dataset.columns) == 0:
        raise HTTPException(status_code=400, detail="The CSV contains no data.")

    app.state.dataset = dataset
    app.state.filename = file.filename
    return _dataset_summary(dataset, file.filename)


@app.get("/dataset")
def get_dataset() -> dict:
    dataset = app.state.dataset
    filename = app.state.filename
    if dataset is None or filename is None:
        raise HTTPException(
            status_code=404,
            detail="No CSV has been uploaded yet. Upload one at /upload-csv.",
        )
    return _dataset_summary(dataset, filename)


@app.post("/run-model")
def run_model() -> dict:
    dataset = app.state.dataset
    if dataset is None:
        raise HTTPException(
            status_code=404,
            detail="Upload a CSV at /upload-csv before running the model.",
        )
    try:
        return _run_churn_model(dataset)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
