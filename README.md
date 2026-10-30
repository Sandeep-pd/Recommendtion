# Customer Churn Model API

A local FastAPI service for uploading a customer churn CSV, inspecting its
contents, and training/evaluating two binary classification models:

- A PyTorch TabTransformer
- An XGBoost classifier

The API returns model evaluation metrics and chart images encoded as
base64-encoded PNG data. It does not require Google Colab; the CSV is uploaded
to the API from your computer.

## Project files

- `api.py` — FastAPI application, CSV upload and preview endpoints, model
  training, evaluation, and chart generation.
- `ibm.py` — the original standalone training script. The API does not import
  or execute this file.

## Requirements

- Python 3.10 or later
- pip
- A modern browser for the interactive API documentation

The API uses FastAPI, Uvicorn, `python-multipart`, pandas, matplotlib,
scikit-learn, PyTorch, and XGBoost. Install them from PowerShell in the project
folder:

```powershell
python -m pip install fastapi uvicorn python-multipart pandas matplotlib scikit-learn torch xgboost
```

PyTorch may offer different installation commands depending on your operating
system and whether you want GPU support. If the command above does not install
a compatible PyTorch build, follow the installation instructions at
[pytorch.org](https://pytorch.org/get-started/locally/).

## Start the API

From the folder containing `api.py`, run:

```powershell
uvicorn api:app --reload
```

By default, the service is available at:

- API base URL: <http://127.0.0.1:8000>
- Interactive Swagger UI: <http://127.0.0.1:8000/docs>
- Alternative ReDoc documentation: <http://127.0.0.1:8000/redoc>

Keep the terminal running while you use the API. Stop the server with
`Ctrl+C`.

## Typical workflow

1. Start the server.
2. Open <http://127.0.0.1:8000/docs>.
3. Expand **POST `/upload-csv`**, select **Try it out**, choose your CSV file,
   and select **Execute**.
4. Confirm the response includes the filename, row count, column names, and a
   preview of the first five rows.
5. Expand **POST `/run-model`**, select **Try it out**, and select **Execute**.
6. Read the metrics in the response. The chart images are included as
   base64-encoded PNG strings.

Upload a new CSV before calling `/run-model` if you want to train on a different
dataset.

## CSV requirements

The CSV must:

- Have a header row.
- Include a column named exactly `Churn` for the label to predict.
- Contain two classes in `Churn`, encoded as `Yes` and `No`, or as `1` and `0`.
  The service also accepts `true` and `false`, case-insensitively.
- Include at least one feature column in addition to `Churn`.
- Have at least two rows in each churn class so the stratified train/test split
  can be made.
- Be no larger than 50 MB.

The Telco Customer Churn dataset commonly uses `customerID`, `TotalCharges`,
and `Churn` columns. If present, `customerID` is removed from the model
features. `TotalCharges` is converted to numeric, and missing numeric values
are filled with the column median. Missing categorical values are represented
as `"Unknown"`. At least one categorical feature column is required.

The service assumes each row represents a customer and each remaining column is
a feature. It does not check whether features are appropriate for prediction
or whether they leak information about the target.

## API endpoints

### `GET /health`

Checks that the service is running.

Example:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Response:

```json
{
  "status": "ok"
}
```

### `POST /upload-csv`

Uploads and parses a CSV file. The upload is stored in the running process's
memory and becomes the active dataset used by `/dataset` and `/run-model`.

Example with `curl`:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/upload-csv" `
  -F "file=@C:\path\to\WA_Fn-UseC_-Telco-Customer-Churn.csv"
```

Example success response:

```json
{
  "filename": "WA_Fn-UseC_-Telco-Customer-Churn.csv",
  "rows": 7043,
  "columns": ["customerID", "gender", "tenure", "TotalCharges", "Churn"],
  "preview": [
    {
      "customerID": "7590-VHVEG",
      "gender": "Female",
      "tenure": 1,
      "TotalCharges": 29.85,
      "Churn": "No"
    }
  ]
}
```

The `preview` above is illustrative; the actual response includes up to five
rows and all dataset columns.

Possible errors:

- `400 Bad Request` — the filename does not end in `.csv`, the CSV is empty or
  invalid, or the file could not be parsed.
- `413 Payload Too Large` — the upload exceeds 50 MB.

### `GET /dataset`

Returns the active dataset's filename, number of rows, columns, and first five
rows. It returns `404 Not Found` until a CSV has been uploaded.

Example:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/dataset
```

### `POST /run-model`

Trains and evaluates both models using the active dataset. It returns
`404 Not Found` if no CSV has been uploaded. Invalid or unsuitable data returns
`400 Bad Request` with a description of the validation problem.

Example:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/run-model
```

The operation performs training synchronously. It may take some time,
especially on a CPU or with a large CSV. The TabTransformer trains for 30
epochs. Both models use a stratified 80/20 train/test split with
`random_state=42`.

The response includes:

- `device` — `cuda` when PyTorch can use a CUDA GPU, otherwise `cpu`.
- `training_rows` and `testing_rows` — row counts for the split.
- `epochs` and `training_loss` — TabTransformer epoch count and loss values.
- `models.transformer` and `models.xgboost` — evaluation results for each
  model. Each includes accuracy, precision, recall, F1, ROC-AUC, a 2-by-2
  confusion matrix, and a classification report.
- `charts.training_loss_png_base64` — PNG of TabTransformer training loss.
- `charts.confusion_matrix_png_base64` — PNG of the TabTransformer confusion
  matrix.

The confusion matrix uses class order `[No Churn, Churn]` for both rows (actual)
and columns (predicted).

To decode a chart in Python:

```python
import base64
import json

with open("model-results.json", encoding="utf-8") as file:
    result = json.load(file)

png_data = base64.b64decode(result["charts"]["training_loss_png_base64"])
with open("training-loss.png", "wb") as file:
    file.write(png_data)
```

Save the `/run-model` JSON response to `model-results.json` before running the
example above. Replace `training_loss_png_base64` with
`confusion_matrix_png_base64` to decode the confusion-matrix image.

## Preprocessing and evaluation details

The API performs the following work when `/run-model` is called:

1. Validates and converts the `Churn` target to binary labels (`No` = 0 and
   `Yes` = 1).
2. Drops `customerID` when present and converts `TotalCharges` to numeric when
   present.
3. Splits data into training and testing sets, preserving the class ratio.
4. Fits an ordinal encoder on categorical training features and applies it to
   both splits. Unseen test categories receive the reserved unknown value.
5. Standardizes numeric features using statistics fitted on the training data.
6. Trains the TabTransformer with weighted binary cross-entropy to account for
   class imbalance.
7. Trains XGBoost using the combined encoded categorical and scaled numeric
   features.
8. Calculates classification metrics on the held-out test data, using a
   probability threshold of 0.5 for class predictions.

ROC-AUC uses predicted probabilities rather than thresholded labels.

## Data handling and limitations

- The uploaded dataset is held only in server memory. It is not written to
  disk, and it is lost when the server restarts.
- Only one active dataset is held per API process. A subsequent upload replaces
  the previous dataset.
- Model objects and preprocessing objects are not persisted by the API.
- Training runs inside the API request and can take time. It is intended for
  local experimentation, not high-concurrency production use.
- There is no authentication or authorization. Do not expose this service to
  untrusted networks or upload data you are not allowed to process.
- Evaluation scores depend on the input data and test split. They are
  experimental measurements, not guarantees about future customers.

## Troubleshooting

### `404`: no CSV has been uploaded

Upload a CSV with `POST /upload-csv` before requesting `/dataset` or
`/run-model`. Uploads are cleared when the API process restarts.

### `400`: CSV needs a `Churn` column

Check that the header is spelled `Churn` with the same capitalization, and
that it contains only the supported binary labels.

### `400`: each churn class needs at least two rows

The service uses a stratified train/test split. Provide at least two examples
for both churn classes; more examples are recommended for meaningful
evaluation.

### `ModuleNotFoundError`

Install the requirements in the same Python environment that runs Uvicorn. For
example:

```powershell
python -m pip install fastapi uvicorn python-multipart pandas matplotlib scikit-learn torch xgboost
```

### Port 8000 is already in use

Run the API on another port:

```powershell
uvicorn api:app --reload --port 8001
```

Then use <http://127.0.0.1:8001/docs>.
