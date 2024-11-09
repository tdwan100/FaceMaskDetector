# Face Mask Detection using Convolutional Neural Networks

This project implements a face mask detection system using a custom Convolutional Neural Network (CNN) model built with TensorFlow and Keras. It classifies individuals as "with mask" or "without mask" in real-time, making it ideal for applications in health and safety compliance, especially during scenarios requiring mask-wearing.

## Features
- **Data Preprocessing**: Converts image data to grayscale and resizes images to a fixed dimension, preparing them as tensors for the CNN model. The processed data is saved in `.npy` format.
- **Custom Dataset Support**: Includes custom "with mask" and "without mask" image datasets. Popular datasets for mask detection can also be incorporated.
- **CNN Model Architecture**: A CNN model with four convolutional layers, dropout regularization, and binary cross-entropy loss for mask classification.
- **Training and Validation**: Uses 90% of data for training and 10% for validation, with model checkpoints to save the best-performing models.
- **Real-time Detection**: Implements real-time face mask detection through webcam using MTCNN for face detection and the trained CNN model for classification.
- **Data Augmentation**: Generates new mask and no-mask images from webcam input to augment the dataset.

## Project Structure
- `dataset/`: Contains the image dataset, organized into folders for "with mask" and "without mask" images.
- `data.npy` & `labels.npy`: Preprocessed image data and labels stored for model training.
- `Face_Mask_Predictor.h5`: Trained model file saved after training completion.
- `Image_Generation.py`: Code for generating mask/no-mask images using webcam.
- `Live_Detection.py`: Code for real-time face mask detection using webcam.

## Getting Started

### Prerequisites
- Python 3.7+
- TensorFlow and Keras
- OpenCV
- MTCNN
- NumPy
- Matplotlib
- Scikit-learn

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Face-Mask-Detection.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Preparation
Place your custom images in the `dataset/with_mask` and `dataset/without_mask` folders or use your own dataset. The preprocessing script will convert these images to grayscale, resize them, and save them in `.npy` format for training.

### Running the Model
1. **Data Preprocessing**: Execute the preprocessing code to generate `data.npy` and `labels.npy`.
2. **Training**: Train the CNN model with the following command:
   ```python
   python Face_Mask_Training.py
   ```
3. **Image Generation**: Run the image generation code to expand the dataset:
   ```python
   python Image_Generation.py
   ```
4. **Live Detection**: Run the live mask detection script to start real-time detection:
   ```python
   python Live_Detection.py
   ```

### Real-Time Face Mask Detection
Once the model is trained, you can run `Live_Detection.py` to detect mask usage in real time using your webcam.

## Results
During training, the model achieved over 99% accuracy on the validation set after 10 epochs. The model saves the best-performing version based on validation loss, allowing for accurate real-time mask detection.

## Model Performance
The model summary:
```
Model: "sequential"
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 150, 150, 32)      320
...
```

## Contributing
Feel free to fork this repository, create a new branch for any feature or bug fix, and submit a pull request. 

## License
This project is licensed under the MIT License.

## Acknowledgments
- [MTCNN](https://github.com/ipazc/mtcnn) for face detection
