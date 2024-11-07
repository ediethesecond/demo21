# Medical Vision Dataset

## Dataset Structure
The dataset is organized into three main directories:

### Raw Data
Contains the original, unmodified data:
- `raw/images/`: Original DICOM files
- `raw/masks/`: Original segmentation masks
- `raw/labels.csv`: Original classification labels

### Processed Data
Contains preprocessed data split into train/val/test sets:
```
processed/
├── train/ (70% of data)
├── val/   (15% of data)
└── test/  (15% of data)
```

Each split contains:
- `images/`: DICOM files
- `masks/`: Segmentation masks in PNG format
- `metadata.csv`: Combined labels and metadata

## Metadata Format
The `metadata.csv` file contains:
```csv
image_id,path,mask_path,label_1,label_2,...,label_14
patient1,images/patient1.dcm,masks/patient1_mask.png,0,1,0,...,1
```

## Data Preprocessing
To preprocess the raw data and create the train/val/test splits:
```bash
python scripts/prepare_data.py --raw_dir data/raw --processed_dir data/processed
```



## Data Statistics
- Total images: X
- Training set: X images
- Validation set: X images
- Test set: X images
- Image size: 224x224
- Number of classes: 14