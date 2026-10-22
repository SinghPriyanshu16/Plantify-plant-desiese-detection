# 🌿 Plantify: Plant Disease Detection System

An AI-driven plant disease diagnosis application based on **DAA-MDeNet** (Dilated Adaptive Attention Multiscale DenseNet). The model accurately detects and classifies 37 different plant leaf diseases across multiple agricultural crops.

---

## 🚀 Key Features
- **DAA-MDeNet Architecture**: Utilizes multiscale dilated convolutions with dilation rates $(1, 2, 4, 8)$ and a Self-Attention mechanism to capture both fine local textures and broad contextual disease patterns.
- **Trained on 37 Disease Classes**: Classifies conditions across Apple, Bell Pepper, Cherry, Citrus, Corn, Grape, Peach, Potato, Strawberry, and Tomato.
- **Interactive Web App**: Powered by Streamlit for instant leaf image uploads, real-time predictions, and top-5 probability rankings.
- **AVLO Optimizer Support**: Implements African Vulture and Lemur Optimizer formulation for hyperparameter exploration.

---

## 🛠️ Project Structure
```text
Plantify-plant-desiese-detection/
├── app.py                      # Interactive Streamlit Web Application
├── models/
│   ├── attention.py            # Self-Attention layer
│   ├── daa_mdenet.py           # DAA-MDeNet model architecture
│   └── daa_mrcnn.py            # Mask R-CNN with custom backbone
├── dataset.py                  # PyTorch Dataset and DataLoader pipeline
├── train.py                    # Training script with Cosine Annealing scheduler
├── avlo.py                     # AVLO optimization algorithm
├── daa_mdenet_best.pth         # Trained model checkpoint weights
├── plant_disease_kaggle.ipynb  # Self-contained Jupyter/Colab notebook
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 💻 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/SinghPriyanshu16/Plantify-plant-desiese-detection.git
cd Plantify-plant-desiese-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Web Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser and upload any plant leaf image.

---

## 📊 Dataset & Model Weights
- **Dataset**: [PlantifyDr Dataset](https://www.kaggle.com/datasets/lavaman151/plantifydr-dataset) (37 classes, 10 crop species).
- **Weights**: Model weights are included in `daa_mdenet_best.pth`.
