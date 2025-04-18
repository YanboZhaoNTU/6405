# FixMatch-Style Semi-Supervised Multi-Label Classification

This project implements a FixMatch-inspired pipeline for semi-supervised multi-label classification using the Bibtex dataset. It supports multi-metric evaluation, masked supervised learning, pseudo-labeling, consistency regularization, and performance visualization.

---

## 🧠 Objective
To build a semi-supervised learning framework that can train multi-label classifiers using only a small fraction of labeled data, leveraging unlabeled data through FixMatch-style pseudo-labeling and consistency loss.

---

## 📁 Dataset
- **Bibtex**: Multi-label dataset in `.arff` format
- Loaded using `scikit-multilearn`
- Located in: `D:/6405/dataset/Bibtex-RandomTrainTest-Mulan/`

---

## 🚦 Pipeline Overview

1. **Load and Mask Labels**
   - 10% of training data is retained with labels
   - The rest are treated as unlabeled (masked with 0s)

2. **PyTorch Dataset & Dataloader**
   - Supports features, labels, and label masks for partial supervision

3. **Model**
   - Simple 2-layer MLP with LayerNorm
   - Outputs multi-label logits for `BCEWithLogitsLoss`

4. **FixMatch-Style Training**
   - Supervised Loss: Binary cross-entropy using masked labels
   - Unsupervised Loss: Pseudo-labels generated from sigmoid(weak_aug) > 0.5
   - Total loss = Supervised + lambda * Unsupervised

5. **Evaluation**
   - 10 multi-label metrics:
     - Micro-F1, Macro-F1, Precision, Recall, Subset Accuracy, Average Precision
     - Hamming Loss, Coverage, Ranking Loss, One-error

6. **Visualization**
   - Training trends for: Micro-F1, Macro-F1, Hamming Loss, Average Precision
   - Saved as `multimetric_curves.png`

---

## 📊 Outputs
- Console logs for each epoch with metrics
- `multimetric_curves.png`: Visualization of 4 key metrics
- (Optional) CSV logging can be added for reproducibility

---

## 🔧 Dependencies
- Python 3.x
- PyTorch
- scikit-learn
- matplotlib
- scikit-multilearn

Install with:
```bash
pip install torch scikit-learn matplotlib scikit-multilearn
```

---

## 🚀 Run Instructions
Ensure Bibtex `.arff` files are in correct path. Then run:
```bash
python ver1.py
```

---

## 🧩 Future Extensions
- Batch processing multiple datasets (yeast, scene, emotions, etc.)
- Save per-epoch metrics to CSV
- Try stronger model backbones (e.g., Transformer)
- Add support for MixMatch or Mean Teacher

