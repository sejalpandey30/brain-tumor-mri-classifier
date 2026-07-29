# brain-tumor-mri-classifier
A deep learning project that uses MRI scans to classify brain tumor into different categories. This repository is built with goal of combining medical imaging and AI to assist in early detection and diagnosis.

Here we have trained to classify the MRI scan into three categories:
- Glioma: A type of tumor that originates in the glial cell of the brain
  ![gradcam_glioma.png](https://github.com/sejalpandey30/brain-tumor-mri-classifier/gradcam_glioma.png)
  
- Meningioma: Tumors that arises from meninges
- Pituitary: Tumor found in the pituitary glands which controls hormones

# Why Classification Matters
Early detection saves lives: Brain tumor often progresses silently. A classifier that can flag abnormalities early gives doctors a chance to intervene before the condition worsens.

Each tumor type requires a different treatment plan. For example, gliomas may need aggressive surgery and radiotherapy, while pituitary tumors might be treated with medication. Misclassification could lead to the wrong treatment.

Radiologists are highly skilled, but fatigue and workload can affect accuracy. AI classifiers act as a second pair of eyes, reducing oversight.

In regions with limited access to specialists, automated classification can help bridge the gap and provide preliminary diagnostic support.

# Features of the model trained

- Image Classification: Detects and classifies tumor types from MRI scans.

- Deep Learning Models: Built using CNN architectures for high accuracy.

- Preprocessing Pipeline: Includes resizing, normalization, and augmentation for robust training.

- Evaluation Metrics: Accuracy, precision, recall, and confusion matrix visualization.

 # Getting Started

1. Clone the repo
```
git clone https://github.com/sejalpandey30/brain-tumor-mri-classifier.git
cd brain-tumor-mri-classifier
```
2. Install dependencies
```
pip install -r requirements.txt
```

# Dataset
The dataset consist of MRI images categorized into 
- Glioma
- Meningioma
- Pituitary
- No Tumor
Dataset source: Kaggle/ public medical imagine dataset

# Result
- Achieved around 87-90% accuracy on test data
- confusion matrix and classification report are included in the notebook

# Contribution
Contributions are welcome! 
Working together communities to create products that matter for everyone!
  - Fork the repo
  - Create a new branch
  - Submit a pull request

# License
This project is licensed under the MIT License
  
