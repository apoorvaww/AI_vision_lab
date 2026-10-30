# AI VisionLab – Automated Image Classification & Model Evaluation

AI VisionLab is a Python-based web application for automated image classification experiments. It allows users to upload image datasets, configure machine learning and deep learning models, train and evaluate multiple architectures, and compare their performance through an interactive interface.

The application follows a structured ML workflow, from dataset discovery and preprocessing to model training, evaluation, comparison, and report generation.

## Features

- **Dataset Upload & Preview:** Upload ZIP files containing image datasets and inspect dataset statistics and class distributions.
- **Automatic Dataset Detection:** Detect common image dataset structures, including ImageFolder and pre-split datasets.
- **Automated Model Selection:** Experiment with multiple classical ML and deep learning architectures.
- **Model Configuration:** Configure training parameters such as epochs and train/test split ratios.
- **Training Progress:** Monitor training progress through an interactive progress interface.
- **Model Evaluation:** Evaluate models using classification metrics and visualizations.
- **Model Comparison:** Compare model performance and identify the best-performing configuration.
- **Exportable Reports:** Generate PDF reports, CSV results, and reproducible Jupyter notebooks.

## Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Machine Learning | Scikit-learn, XGBoost |
| Deep Learning | PyTorch, Torchvision |
| Data Processing | NumPy, Pandas, Pillow |
| Visualization | Matplotlib, Seaborn |
| Model Evaluation | Scikit-learn Metrics |

## Models Supported

### Classical Machine Learning
- Random Forest
- XGBoost
- Support Vector Machine (SVM)
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gradient Boosting

### Deep Learning
- Small CNN
- Deep CNN with Batch Normalization and Dropout

### Transfer Learning
- ResNet18
- MobileNetV2

Transfer learning models use pretrained ImageNet backbones with newly trained classification heads.

## Project Structure

```text
AI_VisionLab/
│
├── app.py
├── run_pipeline.py
├── make_sample_data.py
├── example_config.yaml
├── requirements.txt
├── README.md
├── .gitignore
├── shapes_test.zip
│
├── stage1_pipeline/
│   ├── data.py
│   ├── splits.py
│   ├── models.py
│   ├── search.py
│   └── pipeline.py
│
├── sample_data/
├── static/
├── templates/
├── scratch/
│
└── webapp_data/
    ├── uploads/
    └── results/
```

## Getting Started

Follow these steps to run AI VisionLab locally.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git

cd AI_VisionLab
```

Replace the repository URL with your actual GitHub repository URL.

### 2. Create a Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv venv

.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python3 -m venv venv

source venv/bin/activate
```

### 3. Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

**Note:** Some deep learning models require PyTorch and Torchvision. Make sure your installed versions are compatible with your system and hardware.

### 4. Generate Sample Dataset (Optional)

A synthetic dataset containing colored shapes can be generated for testing.

```bash
python make_sample_data.py
```

If you want to create the ZIP file yourself:

**Windows PowerShell:**

```powershell
Compress-Archive -Path .\sample_data\shapes\* -DestinationPath .\shapes_test.zip -Force
```

**Linux/macOS:**

```bash
cd sample_data
zip -r ../shapes_test.zip shapes
cd ..
```

Alternatively, use the existing `shapes_test.zip` included in the repository.

### 5. Run the Application

Start the Flask application:

```bash
python app.py
```

Open your browser and navigate to:

**http://127.0.0.1:5000/**

You can now upload a dataset and start your image classification experiments.

## Dataset Format

AI VisionLab supports common image classification dataset structures.

### ImageFolder Format

```text
dataset/
│
├── class_a/
│   ├── image1.jpg
│   └── image2.jpg
│
├── class_b/
│   ├── image1.jpg
│   └── image2.jpg
│
└── class_c/
    ├── image1.jpg
    └── image2.jpg
```

Each subfolder represents a separate class.

### Pre-split Dataset Format

The application also recognizes datasets that already contain training, testing, and prediction folders.

Example (Intel Image Classification):

```text
dataset/
│
├── seg_train/
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
│
├── seg_test/
│   ├── buildings/
│   ├── forest/
│   └── ...
│
└── seg_pred/
    ├── image1.jpg
    └── image2.jpg
```

- `seg_train`: Labeled training images.
- `seg_test`: Labeled testing images.
- `seg_pred`: Unlabeled images intended for prediction.

The application recognizes `seg_pred` as an unlabeled prediction dataset rather than treating it as a classification category.

To upload your own dataset, compress the dataset folder into a ZIP archive and upload it through the web interface.

## Application Workflow

The application follows this workflow:

1. Landing Page
2. Dataset Upload
3. Dataset Preview
4. Model Selection
5. Model Configuration
6. Training Progress
7. Results and Evaluation
8. Model Comparison and Report Export

## Running the CLI Pipeline (Optional)

AI VisionLab also provides a command-line interface for running the Stage 1 AutoML pipeline.

```bash
python run_pipeline.py \
    --data sample_data/shapes \
    --out results.json \
    --config example_config.yaml
```

On Windows PowerShell, you can run the command on one line:

```powershell
python run_pipeline.py --data sample_data/shapes --out results.json --config example_config.yaml
```

The pipeline performs dataset validation, generates train/test splits, evaluates candidate models, and saves the resulting configurations and evaluation metrics as JSON.

## Output and Generated Files

The application stores runtime data inside `webapp_data/`.

```text
webapp_data/
│
├── uploads/
│   └── Uploaded datasets
│
└── results/
    └── Training results and generated reports
```

Generated results may include:

- Model evaluation metrics
- Model comparison results
- PDF evaluation reports
- CSV result files
- Reproducible Jupyter notebooks

## Configuration

The `example_config.yaml` file contains the example configuration for the CLI pipeline, including model search parameters.

You can modify this file to experiment with different search configurations.

## Future Improvements

- AI-powered data augmentation recommendations
- Hybrid and ensemble model architectures
- Explainable AI using Grad-CAM and Grad-CAM++
- K-fold cross-validation
- Automated hyperparameter optimization
- Improved training orchestration and experiment tracking

## Contributing

Contributions, suggestions, and bug reports are welcome.

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

## License

This project is available under the MIT License.

---

**AI VisionLab – Making Image Classification Experiments More Accessible.**