# 🧠 SVM Face Recognition

A machine learning project that detects and recognizes faces of political figures using **Support Vector Machine (SVM)** and **Principal Component Analysis (PCA)** with OpenCV.

---

## 👤 Recognizes
- Barack Obama
- Donald Trump
- George W. Bush

---

## 📊 Model Performance

| Metric             | Score  |
|--------------------|--------|
| Accuracy           | 85.29% |
| Mean CV Accuracy   | 83%    |
| Standard Deviation | 0.07   |

### Classification Report

| Person        | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Barack Obama  | 1.00      | 0.71   | 0.83     | 7       |
| Donald Trump  | 0.92      | 0.86   | 0.89     | 14      |
| George W Bush | 0.75      | 0.92   | 0.83     | 13      |

### Confusion Matrix
```
              Predicted
              Obama  Trump  Bush
Actual Obama  [  5     0     2 ]
       Trump  [  0    12     2 ]
       Bush   [  0     1    12 ]
```

---

## 🛠️ Technologies Used

| Library       | Purpose                          |
|---------------|----------------------------------|
| OpenCV        | Face detection & image handling  |
| NumPy         | Array operations                 |
| Scikit-learn  | SVM, PCA, model evaluation       |
| Matplotlib    | Plotting & visualization         |
| Seaborn       | Confusion matrix heatmap         |
| Joblib        | Saving & loading the model       |

---

## 📁 Project Structure

```
SVM Project/
│
├── train_data_2/
│   ├── Barack Obama/         # Training images
│   ├── Donald Trump/         # Training images
│   └── George W Bush/        # Training images
│
├── test_data/                # Test images
│
├── Face_Recognition_model.ipynb       # Main notebook
├── haarcascade_frontalface_default.xml  # Face detector
├── SVM-Face Recognition.sav          # Saved trained model
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

```bash
pip install opencv-python numpy scikit-learn matplotlib seaborn joblib
```

---

## 🔄 How It Works

```
1. Load images from each person's folder
         ↓
2. Detect faces using Haar Cascade Classifier
         ↓
3. Manually approve detected faces (press 'y' to approve)
         ↓
4. Resize approved faces to 50×50 pixels
         ↓
5. Flatten faces from 50×50 → 2500 features
         ↓
6. Reduce features using PCA (2500 → 150)
         ↓
7. Train SVM classifier on reduced features
         ↓
8. Evaluate model — accuracy, report & confusion matrix
         ↓
9. Save trained model as .sav file
```

---

## 🚀 How to Run

**Step 1** — Clone the repository
```bash
git clone https://github.com/Isi2004-star/SVM-Face-Recognition.git
```

**Step 2** — Install dependencies
```bash
pip install opencv-python numpy scikit-learn matplotlib seaborn joblib
```

**Step 3** — Open the notebook
```bash
jupyter notebook Face_Recognition_model.ipynb
```

**Step 4** — Run all cells in order
- During face collection, a window will pop up showing each detected face
- Press **`y`** to approve and add to dataset
- Press **any other key** to skip

---

## 📈 Dataset Summary

| Category      | Samples |
|---------------|---------|
| Barack Obama  | 92      |
| Donald Trump  | 127     |
| George W Bush | 114     |
| **Total**     | **333** |

| Split         | Count |
|---------------|-------|
| Training set  | 299   |
| Testing set   | 34    |

---

## 🧠 Model Architecture

```
Raw Face Image (50×50 grayscale)
        ↓
Flatten → 2500 features
        ↓
Normalize → pixel values 0 to 1
        ↓
PCA → reduce to 150 components (whiten=True)
        ↓
SVM → kernel='rbf'
        ↓
Prediction → Obama / Trump / Bush
```

---

## 📌 Key Concepts

**PCA (Principal Component Analysis)**
Reduces 2500 pixel features down to 150 most important components, removing noise and speeding up training.

**SVM (Support Vector Machine)**
Finds the optimal boundary between each person's facial features using the RBF kernel, which handles non-linear patterns effectively.

**Cross Validation**
Model was validated using 5-fold cross validation achieving a mean accuracy of **83%** with a standard deviation of **0.07**, indicating a stable and consistent model.

---

## 👩‍💻 Author

**Isiri Gangamini Jayanetti**
- GitHub: [@Isi2004-star](https://github.com/Isi2004-star)
