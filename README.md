# Property Address Classification

Classify property addresses into 5 categories using DeBERTa-v3-small.

```
property-address-classification/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── train.csv
│   └── val.csv
│
├── config.py
├── train.py
├── app.py
│
└── results/
    └── (generated files will be saved here)
```

## Categories
- flat
- houseorplot
- landparcel
- commercial unit
- others

## Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Train and evaluate
python train.py

# streamlit file
steamlit run app.py
```

## Results

Results saved in `results/` folder:
- `classification_report.txt` - Performance metrics
- `confusion_matrix.png` - Confusion matrix plot
- `model/` - Trained model

## Model

- **Model**: DeBERTa-v3-small (44M parameters)
- **Accuracy**: See results folder after training
- **Training Time**: ~30-45 minutes on CPU