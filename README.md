# 🎬 Genre-Based Movie Recommendation Using Deep Learning

Recommend movies visually by genre using deep learning and image classification.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Used-red)
![Accuracy](https://img.shields.io/badge/Validation_Accuracy-100%25-success)

---

## 📌 Project Overview

This capstone project aims to develop a **genre-based movie recommendation system** using **Convolutional Neural Networks (CNNs)**. The system classifies movie posters into genre categories and recommends a user-specified number of movies accordingly.

> 🧠 This is an **original project idea** developed from scratch using deep learning concepts and CNNs — no external recommendation systems or pretrained models were used.

---

## 🎯 Objective

To build an image classification model that recommends movies based on genre using CNNs. The model learns genre-specific visual features from posters to make intelligent, user-driven recommendations.

---

## 🗂️ Dataset Description

- **Images Total**: 160  
  - **Training Set**: 80 images  
  - **Validation Set**: 80 images  
- **Genres (8 total)**: Each genre is stored in its own folder inside `train/` and `validation/`.

The dataset is manually organized and loaded using `ImageDataGenerator` with Keras' `flow_from_directory`.

---

## 🧰 Technologies Used

| Component            | Tools/Libraries                 |
|---------------------|----------------------------------|
| Language             | Python                          |
| Frameworks           | TensorFlow, Keras               |
| Platform             | Google Colab (with GPU)         |
| Data Loading         | ImageDataGenerator               |
| Visualization        | Matplotlib, PIL                 |

---

## ⚙️ Data Preprocessing

- **Google Drive Mounted**: Dataset accessed from Drive
- **Image Rescaling**: All images scaled from `[0, 255]` to `[0, 1]`
- **Batched Loading**: Images loaded in batches using `flow_from_directory`
- **Image Size**: All images resized to `150x150`
- **Class Mode**: Categorical (multi-class classification)

---

## 🧠 CNN Model Architecture

| Layer | Description |
|-------|-------------|
| Conv2D + ReLU + MaxPooling | (32, 64, 128 filters) |
| Flatten | Converts 3D features to 1D |
| Dense + Dropout (0.5) | Reduces overfitting |
| Output Layer | Softmax (8 genres) |

> ⚠️ A warning was encountered about setting `input_shape` directly in `Sequential`—this does not affect functionality but can be addressed with best practices.

---

## 🏋️‍♂️ Training

| Setting         | Value               |
|----------------|---------------------|
| Epochs         | 10                  |
| Optimizer      | Adam                |
| Loss Function  | categorical_crossentropy |
| Metric         | Accuracy             |
| Steps per Epoch| 100                  |

---

## 📈 Model Performance

| Metric             | Value         |
|--------------------|---------------|
| Training Accuracy  | ~98.75%       |
| Validation Accuracy| 100.00%       |

> 🧪 The model achieves high training and validation accuracy, though the gap suggests overfitting due to limited data.

---

## 💡 Genre-Based Movie Recommendation Logic

After training, the model can recommend movies based on user input:

### 🔄 How It Works:
1. **User Inputs**: Genre + number of movies desired  
2. **Random Images Displayed**: From the selected genre's folder  
3. **Each Image**:  
   - Resized to `150x150`  
   - Displayed with its genre label  
4. **Clean Output**: Only images and titles shown — no extra messages unless there's an error

> 🎥 This creates a visual, genre-based movie discovery experience for the user.

---

## 📌 Result Analysis

- Efficient genre filtering
- Visually engaging outputs
- Custom results per user request
- Purely built using CNNs, not external libraries or APIs

---

## ✅ Conclusion

This project successfully uses **Convolutional Neural Networks** to classify and recommend movies by genre. By training on poster images, the system provides a visually intuitive interface for discovering genre-specific content, based entirely on learned visual patterns.

---

echo "## 🚀 Future Enhancements

This project serves as a strong foundation for deep learning-based content recommendation. With additional time, resources, and data, the following enhancements are planned:

- **🎞️ Larger Dataset Integration**  
  Incorporate thousands of movie posters across more diverse genres (e.g., Romance, Thriller, Documentary) to improve model generalization and reduce overfitting.

- **🌐 Live Web Application**  
  Build a fully functional, cloud-hosted web interface using **Streamlit** or **Flask** where users can:
  - Select a genre
  - Specify how many recommendations they want
  - View poster previews with genre predictions in real-time

- **🧠 Model Improvements**  
  - Fine-tune with pre-trained models like **ResNet50 or EfficientNet** for better feature extraction.  
  - Add **ensemble learning** to improve prediction confidence.

- **📊 Explainability & Insights**  
  - Add Grad-CAM visualizations to highlight what parts of a poster influenced genre classification.  
  - Visual dashboard showing genre distribution and model confidence scores.

- **📱 Mobile Compatibility**  
  Optimize the system for use on mobile devices or as a lightweight PWA (Progressive Web App).

- **📂 Content-Based Recommendations**  
  Extend beyond genre classification by integrating metadata (e.g., IMDb ratings, cast, plot) for smarter hybrid recommendations.
" >> README.md

## 🧑‍💻 Author

**TAnna Naga Sri Durga Mallesh**  
B.Tech – Computer Science and Engineering  
Sir C R Reddy College of Engineering, Eluru

---

## 📜 License

This project is licensed under the **MIT License** – you may use or modify it with attribution.

---

## 🙏 Acknowledgments

Special thanks to:

- **Mr. G. Satya Narayana Sir**, Assistant Professor, CSE Department, Sir C R Reddy College of Engineering – for continuous Deep learning Guidance and Support
