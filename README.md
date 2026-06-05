# 🧠 ZubNet-A: Lightweight CNN for Brain Tumor Classification

ZubNet-A is a lightweight and efficient convolutional neural network (CNN) designed for accurate multi-class brain tumor classification using contrast-enhanced MRI (CE-MRI) images.

This repository provides the implementation of the proposed model described in our research work.

---

## 🚀 Key Features

- ⚡ Lightweight architecture (only **0.29M parameters**)
- 🧠 Designed specifically for **MRI-based brain tumor classification**
- 🔍 Incorporates **Depthwise Separable Convolutions**
- 🎯 Uses **Squeeze-and-Excitation (SE) Attention Mechanism**
- 🧩 Employs **Stage-wise Feature Extraction**
- 🛡️ Patient-level data splitting (prevents data leakage)

---

## 📊 Performance

| Metric        | Value |
|--------------|------|
| Accuracy     | **96.98%** |
| Precision    | ~0.96 |
| Recall       | ~0.97 |
| F1-score     | ~0.97 |
| Specificity  | ~0.98 |

---

## 🧠 Classification Categories

The model performs multi-class classification across:

- Glioma
- Meningioma
- Pituitary Tumor

---

## 📂 Dataset

- Source: **Figshare Brain Tumor Dataset (CE-MRI)**
- Total Samples: 3064 MRI slices
- Patients: 233
- Data split: **Patient-level (80% training / 20% testing)**

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

## 🧪 Model Details
- Framework: PyTorch
- Loss Function: Categorical Cross-Entropy
- Batch Size: 32
- Epochs: 50
- Learning Rate: 0.001
- Scheduler: StepLR (0.5 every 10 epochs)

## 📈 Results Visualization
<img width="1112" height="454" alt="curves" src="https://github.com/user-attachments/assets/f704b6bf-bfeb-40e7-8b0d-96cb9c478a4d" />
<img width="508" height="477" alt="zubnet-ACM" src="https://github.com/user-attachments/assets/1ab597eb-4472-41bd-a6d6-897eebf974cc" />

## 🔗 Reproducibility
To ensure transparency and reproducibility, all implementation details and experiments are provided in this repository.

## 👨‍💻 Author
1- Alzubair Alqaraghuli
- PhD Student – Computer Science
- Specialization: Deep Learning & Medical Imaging

2- Serkan Ozturk

## 📜 License
This project is licensed under the MIT License.
