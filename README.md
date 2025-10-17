# 🩸 Malaria Cell Image Classification using Deep Learning

## 📘 Overview

This project uses **Convolutional Neural Networks (CNNs)** to classify malaria-infected and uninfected cell images. It aims to support medical diagnosis by providing an AI-driven image classification tool that can detect **malaria parasites in blood cell images** with high accuracy.

The trained model is integrated with a **Gradio UI** for interactive testing and visualization, allowing users to upload images and get real-time predictions.

---

## 🚀 Features

* 🧠 **Deep Learning-based CNN model** for malaria cell detection
* 🖼️ **Real-time image classification** via an interactive **Gradio web interface**
* 📊 Preprocessing pipeline with **data augmentation and normalization**
* 🧰 Built using **TensorFlow/Keras, NumPy, Matplotlib, and Gradio**
* 🎨 Clean UI with soft purple-blue theme for better user experience

---

## 📂 Dataset

The dataset used is the **Malaria Cell Images Dataset** from [Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria).
It contains two folders:

* `Parasitized/` – Images of cells infected with malaria parasites
* `Uninfected/` – Images of healthy cells

---

## 🧩 Model Architecture

The CNN model consists of:

* **3 Convolutional Layers** with ReLU activation and MaxPooling
* **Flatten + Dense Layers** for feature extraction
* **Sigmoid Output Layer** for binary classification

The model is trained using **Binary Crossentropy Loss** and **Adam Optimizer**.

---

## ⚙️ Technologies Used

* **Python 3.x**
* **TensorFlow / Keras**
* **NumPy, Matplotlib, PIL**
* **Gradio**
* **Google Colab / Jupyter Notebook**

---

## 🖥️ How to Run

### Step 1: Clone the Repository

```bash
git clone https://github.com/YourUsername/Malaria-Cell-Classification.git
cd Malaria-Cell-Classification
```

### Step 2: Install Dependencies

```bash
pip install tensorflow gradio numpy matplotlib pillow
```

### Step 3: Run the Notebook

Upload the dataset and execute the notebook `malaria_classification.ipynb` to train and save the model.

### Step 4: Launch the Gradio App

```bash
python app.py
```

Then open the local Gradio link in your browser.

---

## 🧪 Sample Prediction

Upload a microscopic image of a blood cell, and the model will output:

* **Parasitized** 🦠
* **Uninfected** ✅
  along with the confidence score.

---

## 📈 Results

* Achieved **accuracy ~95%** on validation data
* Effectively distinguishes between healthy and infected cells
* Can be improved further with transfer learning (e.g., using VGG16, ResNet50)

---

## 💡 Future Enhancements

* Deploy the model using **Hugging Face Spaces** or **Streamlit Cloud**
* Add **Grad-CAM** visualization for explainable AI insights
* Integrate a **database (Supabase / MongoDB)** to store user test history
* Expand dataset for multi-class classification of parasite stages

---

## 👩‍💻 Author

**Anjum Khan**
💻 Passionate about AI + Healthcare Innovation

---

## 🧠 Acknowledgements

* Dataset: [Kaggle - Malaria Cell Images Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
* Tools: TensorFlow, Keras, Gradio
* Inspired by research on AI-driven diagnostic systems in healthcare
