# Implementing Authentic Counterhate Arguments

This repository contains a project replicating and extending the [EMNLP_23](https://2023.emnlp.org/) paper "Finding Authentic Counterhate Arguments: A Case Study with Public Figures". Authors: Abdullah Albanyan, Ahmed Hassan, and Eduardo Blanco, focusing on the identification and validation of counterhate arguments against individual-targeted online hate speech. 

This project is part of the "Reproducibility challenges in research papers" and is further documented in our reproducibility study report, which can be accessed [here](./Counterhate_Arguments_Report.pdf) by Reproducibility: Sweta Pati and Swabhi Papneja.


## 🚀 Getting Started

Follow these steps to set up and run the project:

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/swetapati22/Counterhate_Arguments.git
```

### **2️⃣ Navigate to the Project Directory**
```sh
cd Counterhate_Arguments
```

### **3️⃣ Install Dependencies**
Make sure you have Python installed. Then, run:
```sh
pip install -r requirements.txt
```

## 🔄 Data Preparation

### **4️⃣ Run Data Preparation Script**
This script loads and preprocesses the input data for the model.
```sh
python prepare_data.py --csv-file <path_to_csv_file> --level <level> --output-dir <output_directory>
```
Where:
- `--csv-file`: Path to the `.csv` data file.
- `--level`: Either `paragraph` or `article`.
- `--output-dir`: Directory to save the processed data (default: `./Dataloaders/`).

#### **Example Commands:**
For Article:
```sh
python prepare_data.py --csv-file ../Data/articles.csv --level article --output-dir ../Dataloaders/
```
For Paragraph:
```sh
python prepare_data.py --csv-file ../Data/paragraphs.csv --level paragraph --output-dir ../Dataloaders/
```

After running this, a `Dataloaders/` folder will be created containing:
- `train.pth` (Training set)
- `valid.pth` (Validation set)
- `test.pth` (Test set)

## 🎯 Model Training

### **5️⃣ Train the Model**
```sh
python train.py --data-dir <processed_data_path> --level <level> --output-dir <output_path>
```
Where:
- `--data-dir`: Directory of the processed data (default: `./Dataloaders/`).
- `--level`: Either `paragraph` or `article`.
- `--output-dir`: Directory to save the trained model (default: `./Output/`).

#### **Example Commands:**
For Article:
```sh
python train.py --data-dir ../Dataloaders/ --level article --output-dir ../Output/
```
For Paragraph:
```sh
python train.py --data-dir ../Dataloaders/ --level paragraph --output-dir ../Output/
```

## 📊 Model Evaluation

### **6️⃣ Evaluate the Model**
```sh
python test.py --data-dir <processed_data_path> --trained-model-dir <trained_model_path> --output-dir <output_path>
```
Where:
- `--data-dir`: Directory of processed data (default: `./Dataloaders/`).
- `--trained-model-dir`: Directory of trained models (default: `./Output/`).
- `--output-dir`: Directory to save the predictions (default: `./Output/`).

#### **Example Commands:**
For Article:
```sh
python test.py --data-dir ../Dataloaders/ --trained-model-dir ../Output/ --output-dir ../Output/
```
For Paragraph:
```sh
python test.py --data-dir ../Dataloaders/ --trained-model-dir ../Output/ --output-dir ../Output/
```

## 🛠 Error Analysis

### **7️⃣ Perform Error Analysis**
```sh
python error_analysis.py --data-dir <processed_data_path> --trained-model-dir <trained_model_path> --output-dir <output_path>
```
#### **Example Commands:**
For Article:
```sh
python error_analysis.py --data-dir ../Dataloaders/ --trained-model-dir ../Output/ --output-dir ../Output/
```
For Paragraph:
```sh
python error_analysis.py --data-dir ../Dataloaders/ --trained-model-dir ../Output/ --output-dir ../Output/
```

## ⚡ Quick Run Scripts
To simplify execution, you can run the following scripts instead of manually running all steps:
- `article_script.sh` – Automates processing, training, and evaluation for **articles**.
- `paragraph_script.sh` – Automates processing, training, and evaluation for **paragraphs**.

## 📜 Citation
If you use this work, please cite: (Our reproducibility work)
```
@misc{papneja_pati_2024,
  author = {Papneja, S. and Pati, S.},
  title = {Implementing Authentic Counterhate Arguments},
  year = {2024},
  url = {https://github.com/swetapati22/Counterhate_Arguments},
  note = {GitHub Repository}
}
```

### **🔍 Original Research Paper:**
```
@inproceedings{albanyan-etal-2023-finding,
    title = "Finding Authentic Counterhate Arguments: A Case Study with Public Figures",
    author = "Albanyan, Abdullah  and Hassan, Ahmed  and Blanco, Eduardo",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.855",
    doi = "10.18653/v1/2023.emnlp-main.855",
    pages = "13862--13876",
}
```
