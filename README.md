# VidFormer Codebase

This repository contains the core code for VidFormer-based remote photoplethysmography (rPPG) experiments, including dataset preprocessing, model definition, and training scripts.

## Project Structure

- `data_preprocess/`
  - `unified_dataset_split.py`: unified dataset splitting script for `UBFC`, `PURE`, `COHFACE`, `ECG-fitness`, `DEAP`, and `MMPD`
  - `Dataset_Base.py`: dataset loading utilities used during training
- `Module_VidFormer/`
  - `Main.py`: main training script
- `VidTransformer/`
  - VidFormer model definitions
- `Util/`
  - helper modules used by the network

## Requirements

The code depends on common scientific Python packages and several project-specific libraries:

- `audtorch==0.6.4`
- `dlib==19.24.0`
- `einops==0.8.2`
- `h5py==3.8.0`
- `heartpy==1.2.7`
- `matplotlib==3.7.1`
- `numpy==1.24.2`
- `opencv_python==4.8.1.78`
- `Pillow==12.2.0`
- `pywin32==306`
- `scipy==1.17.1`
- `skimage==0.0`
- `thop==0.1.1`
- `timm==1.0.27`
- `torch==1.13.1`
- `torch==2.2.2`
- `torchvision==0.14.1+cu117`
- `torchvision==0.17.2`
- `tqdm==4.65.0`


You also need the dlib landmark file:

- `source/shape_predictor_81_face_landmarks.dat`

## Dataset Preprocessing

Use the unified preprocessing script to split raw data into training samples.

Example:

```bash
python data_preprocess/unified_dataset_split.py --dataset pure --root YOUR_PURE_ROOT
```

Other examples:

```bash
python data_preprocess/unified_dataset_split.py --dataset ubfc --root YOUR_UBFC_ROOT
python data_preprocess/unified_dataset_split.py --dataset cohface --root YOUR_COHFACE_ROOT
python data_preprocess/unified_dataset_split.py --dataset ecg-fitness --root YOUR_ECG_ROOT --bbox-root YOUR_BBOX_ROOT
python data_preprocess/unified_dataset_split.py --dataset mmpd --root YOUR_MMPD_ROOT --data-key video --label-key GT_ppg
```

## Training

Run the main training script with the prepared dataset root:

```bash
python Module_VidFormer/Main.py --dataset-root YOUR_DATASET_ROOT
```

Example:

```bash
python Module_VidFormer/Main.py --dataset-root F:\LJC\PURE\PURE_unzip --dataset-name PURE --data-key video --label-key GT_ppg
```

The training script saves:

- `final_model.pt`
- `train_config.json`
- `history.json`
- `final_metrics.json`
- training curves as `.png`

## Notes

- The training script is configured to perform training for all epochs first and run evaluation only once at the end.
- Paths in the examples should be replaced with your local dataset locations.
- For IDE usage, you can either pass command-line arguments in the run configuration or set default paths directly in the script.
