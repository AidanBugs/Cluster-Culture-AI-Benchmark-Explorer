# Cluster & Culture — AI Benchmark Explorer

An interactive web application for exploring and classifying Large Language Model (LLM) benchmarks. This tool provides visualizations and analysis of AI model performance data through PCA (Principal Component Analysis), custom-axis scatter plots, and machine learning classification.

## Features

- **PCA Explorer**: Visualize benchmark data in 2D or 3D using Principal Component Analysis
- **Custom Axes**: Create scatter plots with user-defined axes from benchmark scores
- **Classification**: Apply machine learning models (Logistic Regression, SVM, Decision Tree) to classify models
- **Interactive UI**: Web-based interface with dynamic plotting using Plotly.js

## Data

The application uses LLM benchmark data including scores from various evaluation datasets like BBH, GPQA, IFEval, MATH, MMLU-PRO, and MUSR. Data includes model metadata such as company, origin, and release dates.

## Prerequisites

- Python 3.8+
- pip

## Installation

1. Clone or download this repository
2. Navigate to the project directory
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

Open your web browser and go to `http://localhost:8000`

## Project Structure

- `main.py`: FastAPI backend server
- `requirements.txt`: Python dependencies
- `data/`: Benchmark data files and metadata
- `static/`: Frontend HTML, CSS, and JavaScript files

## Technologies Used

- **Backend**: FastAPI, Python
- **Data Processing**: pandas, NumPy, scikit-learn
- **Frontend**: HTML, CSS, JavaScript, Plotly.js
- **Visualization**: PCA, scatter plots, ML classification</content>
<parameter name="filePath">c:\Users\aidan\Documents\Coding\CWRU\HTEC301\README.md