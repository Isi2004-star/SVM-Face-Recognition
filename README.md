# 🧠 SVM Face Recognition

A machine learning project that recognizes faces using 
Support Vector Machine (SVM) and Principal Component Analysis (PCA).

## 👤 Recognizes
- Barack Obama
- Donald Trump
- George W. Bush

## 🛠️ Technologies Used
- Python
- OpenCV (face detection)
- Scikit-learn (SVM + PCA)
- NumPy
- Matplotlib / Seaborn

## 📁 Project Structure

```
SVM Project/
│
├── train_data_2/
│   ├── Barack Obama/
│   ├── Donald Trump/
│   └── George W Bush/
│
├── test_data/
│   ├── barack-obama.jpeg
│   ├── trump.jpg
│   └── ...
│
├── Face_Recognition_model.ipynb
├── haarcascade_frontalface_default.xml
├── SVM-Face Recognition.sav
├── .gitignore
└── README.md
```

## 🚀 How to Run
1. Place training images in `train_data_2/` folder
2. Run all cells in `Face_Recognition_model.ipynb`
3. Press `y` to approve detected faces during collection
4. Model trains automatically and saves as `.sav` file
