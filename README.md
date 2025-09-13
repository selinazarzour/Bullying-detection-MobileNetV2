# Bullying Detection with MobileNetV2

## Project Overview

This project presents a **video-based bullying detection system** using deep learning, tailored for real-life scenarios. Bullying is a critical issue in schools, workplaces, and social environments, and automated detection can help prevent harm and enable timely intervention. We leverage **MobileNetV2** and **LSTM** architectures to classify video sequences as "Bullying" or "NonBullying" with high accuracy, even on a small, personalized dataset.

## Objective

- **Automate the detection of bullying in videos**: Accurately identify incidents of bullying from video feeds.
- **Demonstrate deep learning skills**: Fine-tune state-of-the-art models and build end-to-end ML pipelines.
- **Showcase technical breadth**: Use modern frameworks, data engineering, transfer learning, and model evaluation techniques.

## Dataset

- **Source**: [Real-Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)
- **Classes**: NonBullying and Bullying
- **Sample Size**: 20 videos per class for demonstration purposes (customizable for larger datasets)

## Frameworks & Technologies

- **Python 3.7**
- **TensorFlow / Keras**: Deep Learning model training and architecture
- **OpenCV**: Video frame extraction and preprocessing
- **Scikit-learn**: Dataset splitting, metrics, and evaluation
- **Matplotlib / Seaborn**: Data visualization and model performance plots
- **Google Colab / Kaggle Notebooks**: Interactive experimentation environment

## Model Architecture

### Feature Extraction

- **MobileNetV2 (pre-trained on ImageNet)**: Efficient CNN for spatial feature extraction from video frames.
- **Fine-Tuning**: Last 40 layers of MobileNetV2 are trainable to adapt to bullying detection.

### Sequence Modeling

- **TimeDistributed Layer**: Applies MobileNetV2 to each frame in the video sequence.
- **Bidirectional LSTM**: Captures temporal dynamics in both forward and backward directions.
- **Dense Layers & Dropout**: Deep classification layers with regularization to prevent overfitting.

### Complete Pipeline

1. **Video Frame Extraction**: Resize frames to 64x64 and normalize pixel values.
2. **Feature Extraction**: MobileNetV2 processes each frame.
3. **Temporal Modeling**: LSTM layers learn temporal relationships.
4. **Classification**: Dense layers predict probabilities for Bullying/NonBullying.

## Training & Evaluation

- **Train/Test Split**: 80/20 for robust evaluation on unseen samples.
- **Early Stopping & ReduceLROnPlateau**: Prevent overfitting and stabilize learning.
- **Metrics**: Accuracy, confusion matrix, and classification report for transparency.
- **Visualization**: Training curves for loss and accuracy, confusion matrix heatmaps.

## Results

- **Accuracy**: Achieved strong accuracy on the test set, demonstrating feasibility even with a small dataset.
- **Model Interpretability**: Results visualized for clear recruiter presentation.
- **Scalability**: Approach extensible to larger datasets and real-world applications.

## Skills Demonstrated

- **Data Engineering**: Automated video-to-frame pipelines and dataset management.
- **Transfer Learning**: Fine-tuning state-of-the-art CNNs for custom tasks.
- **Sequence Modeling**: LSTM for temporal understanding in videos.
- **Model Evaluation**: Use of best practices for validation and performance analysis.
- **End-to-End ML Solution**: From raw videos to actionable predictions and insights.
- **Collaboration**: Modular, readable code with clear documentation for teamwork and future development.

## How to Run

1. **Install requirements**: `pip install tensorflow keras opencv-python scikit-learn matplotlib seaborn`
2. **Download datasets**: Place datasets in the specified folder structure as in the notebook.
3. **Execute notebook**: Run each cell in `Bullying_detection_project.ipynb` to preprocess data, train models, and visualize results.

## Next Steps

- **Expand dataset**: Incorporate more diverse and larger datasets for improved generalization.
- **Deploy model**: Integrate with real-time video monitoring for production use.
- **Explore advanced architectures**: Test 3D CNNs, transformers, and attention mechanisms.
- **Ethics & Privacy**: Ensure responsible use, privacy protection, and bias mitigation.

## Contact

For questions, collaborations, or demo requests, feel free to reach out via GitHub or [LinkedIn](https://www.linkedin.com/in/your-profile).

---

*This project exemplifies practical deep learning for video understanding, and demonstrates engineering skills highly relevant to AI/ML roles in industry and academia.*
