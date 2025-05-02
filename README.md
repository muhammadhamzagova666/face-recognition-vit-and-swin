# Face Recognition using ViT and Swin Transformers

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Description

This repository implements and compares two transformer-based face recognition pipelines:
1. **Swin Transformer** – A hierarchical vision transformer yielding state-of-the-art performance in facial feature extraction.  
2. **Vision Transformer (ViT)** – A non-hierarchical transformer trained from scratch for face recognition.

Our goal is to demonstrate the superior representational power of Swin over vanilla ViT on the Labelled Faces in the Wild (LFW) dataset, and to analyse the trade-offs in training complexity, inference speed, and recognition accuracy.

## Table of Contents

- [Features](#features)  
- [Dataset](#dataset)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Preparing the Data](#preparing-the-data)  
  - [Training Swin Model](#training-swin-model)  
  - [Training ViT Model](#training-vit-model)  
  - [Evaluation](#evaluation)  
- [Results](#results)  
- [Comparison & Discussion](#comparison--discussion)  
- [Pros & Cons](#pros--cons)  
- [Contributing](#contributing)  
- [License](#license)  
- [Contact](#contact)  

## Features

- End-to-end training pipelines for both Swin Transformer and ViT on LFW  
- Modular PyTorch implementation with configurable hyperparameters  
- Automated evaluation scripts computing accuracy, ROC curves, and confusion matrices  
- Jupyter notebooks demonstrating experiments and visualisations  

## Dataset

We use the Labelled Faces in the Wild (LFW) dataset, available on Kaggle:

[Labelled Faces in the Wild (LFW) Dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)

Download and unpack into:
```

data/lfw-deepfunneled/
└── lfw-deepfunneled/
├── ...
│   ├── ...
│   └── ...
└── ...

````

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/muhammadhamzagova666/face-recognition-vit-and-swin.git
   cd face-recognition-vit-and-swin
   ```

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify installation**

   ```bash
   python -c "import torch; import transformers; print('Setup OK')"
   ```

## Usage

### Preparing the Data

```bash
# Ensure Kaggle CLI is configured with your API token
kaggle datasets download -d jessicali9530/lfw-dataset
unzip lfw-dataset.zip -d data/lfw-deepfunneled
```

### Training Swin Model

```bash
python swin-lfw.py \
  --data-dir data/lfw-deepfunneled \
  --output-dir experiments/swin \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 1e-4
```

### Training ViT Model

```bash
python vit-lfw.py \
  --data-dir data/lfw-deepfunneled \
  --output-dir experiments/vit \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 1e-4
```

## Results

The trained models can be found on this link:

[Face Recognition Trained Models using ViT and Swin Transformers on LFW Dataset](https://drive.google.com/drive/folders/1hbdr1LVDz2pHtKFv55q0Lo3KviPJVt1t?usp=drive_link)

| Model         | Val Accuracy (%) | Train Accuracy (%) | Loss |
| ------------- | ---------------- | ------------------ | ---- |
| Swin          | 91.4             | 98.2               | 1.19 |
| ViT (scratch) | 81.1             | 93.4               | 1.49 |

*Key takeaway:* Swin Transformer outperforms ViT by a substantial margin in both recognition accuracy and ROC AUC, while also delivering faster inference due to its hierarchical design.

## Comparison & Discussion

* **Architectural Differences**

  * *Swin*: Window-based multi-scale self-attention; strong locality inductive bias.
  * *ViT*: Global self-attention on fixed-size patches; requires more data to generalize.

* **Training Complexity**

  * Swin converges in \~25 epochs vs. ViT’s \~80 epochs for comparable performance.
  * ViT demands larger batch sizes and more careful learning-rate scheduling.

* **Inference Speed**

  * Swin’s hierarchical tokens reduce per-layer complexity, yielding \~30% faster inference.

## Pros & Cons

| Model | Pros                                                                      | Cons                                                                |
| ----- | ------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Swin  | • Superior accuracy<br>• Faster inference<br>• Robust to scale variations | • More complex implementation<br>• Slightly larger memory footprint |
| ViT   | • Simpler architecture<br>• Easier to customize for new modalities        | • Lower accuracy<br>• Slower converge and inference                 |

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a pull request

Please ensure the code passes linting and all tests before merging.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or feedback, please open an issue or contact:

* **Maintainers**:
  - [Muhammad Hamza](https://github.com/muhammadhamzagova666/)
  - [Emmanuel](https://github.com/emmanuelmoon/)
  - [Jatin Kesnani](https://github.com/Jatin-Kesnani/)

* **Project URL**: [Face Recognition using ViT and Swin Transformers](https://github.com/muhammadhamzagova666/face-recognition-vit-and-swin/)
