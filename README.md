# 👁️ Diabetic Retinopathy Detection 🩺

This project aims to detect diabetic retinopathy from retinal images using deep learning and image processing techniques. The solution leverages convolutional neural networks (CNNs) and various preprocessing steps to classify the severity of diabetic retinopathy, supporting early diagnosis and treatment. 🧑‍⚕️

## 🎯 Purpose
Diabetic retinopathy is a leading cause of blindness among diabetic patients. Early detection through automated image analysis can help in timely intervention. This project provides a pipeline for preprocessing, training, and predicting diabetic retinopathy stages from retinal images. 👁️

## ⚙️ Installation
1. Clone the repository:
	```bash
	git clone https://github.com/GriselQ23/diabetic-retinopathy-detection.git
	cd diabetic-retinopathy-detection
	```
2. Install the required dependencies:
	```bash
	pip install -r requirements.txt
	```

## 🚀 Usage
- The main workflow is organized in Jupyter notebooks:
	- `Pre_process2.ipynb`: Preprocesses the retinal images for model training and inference. 🖼️
	- `Idrid_v2.ipynb`: Contains model training and evaluation code. 🧠
	- `Prediccion.ipynb`: Performs predictions on new images using the trained model. 🔮

To run the notebooks:
1. Open the desired notebook in Jupyter:
	 ```bash
	 jupyter notebook
	 ```
2. Follow the instructions in each notebook cell to preprocess data, train the model, or make predictions. 👍

## 📦 Dependencies
The main dependencies are listed in `requirements.txt` and include:
- numpy
- scipy
- pillow
- matplotlib
- scikit-image
- scikit-learn
- opencv-python
- imgaug
- imageio
- networkx
- shapely
- tensorflow
- keras
- google-colab (for Colab usage)

For a full list, see the `requirements.txt` file. 📄

## 📚 Notebooks: Technologies and Architectures

### 🖼️ Pre_process2.ipynb
- **Technologies Used:**
	- Python libraries: `os`, `cv2` (OpenCV), `matplotlib`, `random`, `imageio`, `imgaug`
	- Google Colab for cloud-based execution and data access
	- Data augmentation: The `imgaug` library is used to apply transformations such as rotation, scaling, and flipping to increase dataset diversity.
	- Image preprocessing: OpenCV (`cv2`) is used for image reading, resizing, and color space conversion. `matplotlib` is used for visualization.
- **Purpose:**
	- Prepares the retinal images for model training. Includes steps for mounting Google Drive, extracting datasets, visualizing random images, counting files, and applying augmentation. 🧑‍💻

### 🧠 Idrid_v2.ipynb
- **Technologies Used:**
	- Deep learning: `tensorflow.keras` for model building, training, and evaluation
	- Image processing: `cv2`, `matplotlib`
	- Data handling: `os`, `random`, `shutil`
	- Google Colab for data access
- **Architecture:**
	- **U-Net:** A convolutional neural network for image segmentation. U-Net consists of an encoder (downsampling path), a bottleneck, and a decoder (upsampling path) with skip connections between corresponding layers.
		- Encoder: Stacks of convolutional and max-pooling layers to capture context.
		- Bottleneck: Deepest part of the network with the most feature channels.
		- Decoder: Upsampling and concatenation with encoder features for precise localization.
		- Output: Final convolutional layer with sigmoid activation for segmentation masks.
- **Purpose:**
	- Training and evaluating the U-Net model for segmenting retinal images, a crucial step for automated diabetic retinopathy analysis. 🩺

### 🔮 Prediccion.ipynb
- **Technologies Used:**
	- Deep learning: `tensorflow.keras` for loading trained models and making predictions
	- Image processing: `cv2`, `numpy`
	- Evaluation: `sklearn.metrics` for confusion matrix, `LabelEncoder` for label handling
	- Visualization: `matplotlib`
- **Purpose:**
	- Loads a trained model and applies it to new images for prediction. Processes images, encodes labels, performs predictions, and visualizes results using a confusion matrix. 📊

---

| 📒 Notebook         | 🛠️ Main Technologies                | 🏗️ Main Architecture | 🎯 Purpose                                    |
|---------------------|----------------------------------|-------------------|---------------------------------------------|
| 🖼️ Pre_process2     | OpenCV, imgaug, matplotlib       | N/A               | Preprocessing, augmentation, visualization  |
| 🧠 Idrid_v2         | TensorFlow/Keras, OpenCV         | U-Net             | Segmentation model training & evaluation    |
| 🔮 Prediccion       | TensorFlow/Keras, OpenCV, sklearn| Trained model     | Prediction & evaluation on new data         |

For a full list of dependencies, see the `requirements.txt` file. 😊

## 📝 License
This project is for educational and research purposes. 📚