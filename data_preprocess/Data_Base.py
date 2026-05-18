from pathlib import Path
import random
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dlib
import cv2 as cv
import h5py
import heartpy
import re

class Dataset(Dataset):
    """
    Main modifications compared with the original version:
    1. Replace global per-clip z-score normalization with Log-Relative Reflectance Normalization (LRRN).
    2. Fix horizontal/vertical flip dimensions for [T, H, W, C] video tensors.
    3. Add optional illumination/color augmentation before normalization.
    4. Make temporal resampling more robust to boundary cases.
    5. Avoid persistent_workers=True when num_workers=0 in build_dataloaders().
    """

    def __init__(
            self,
            subject_dirs,
            data_key,
            label_key,
            split_number,
            window_length,
            is_train,
            Dataset_name='UBFC',
            fixed_length=250,
            sample_rate=30
    ):
        self.dataset=Dataset_name
        self.data_key = data_key
        self.label_key = label_key
        self.split_number = split_number
        self.window_length = window_length
        self.is_train = is_train
        self.fixed_length = fixed_length
        self.sample_rate = sample_rate
        self.samples = []

        if self.dataset=='MMDP':
            for subject_dir in subject_dirs:
                mat_files = sorted(Path(subject_dir).glob("*.npy"))
                for mat_path in mat_files:
                    if "_label" in mat_path.name:
                        continue
                    self.samples.append(mat_path)
        elif self.dataset=='DEAP':
            for subject_dir in subject_dirs:
                mat_files = sorted(Path(subject_dir).rglob("*.avi"))
                for mat_path in mat_files:
                    self.samples.append(mat_path)
        else:
            for subject_dir in subject_dirs:
                mat_files = sorted(Path(subject_dir).rglob("*.avi"))
                for mat_path in mat_files:
                    self.samples.append(mat_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        mat_path = self.samples[item]
        if self.dataset=='MMDP':
            label_path = mat_path.with_name(mat_path.name.replace('_data', '_label'))
            x = np.load(mat_path, mmap_mode='r')[:self.fixed_length, ...]
            label = np.load(label_path, mmap_mode='r')[0, :self.fixed_length]
            x = torch.tensor(x.copy(), dtype=torch.float32)
        elif self.dataset=='DEAP':
            label_path=str(mat_path).replace('face_video', 'data_original').replace('.avi', '.txt')
            label_f = open(label_path, 'r')
            label = label_f.readlines()
            label = ''.join(label)
            label = label.replace('[', '').replace(']', '').replace(',', ' ').split()
            label = list(map(float, label))
            face_roi = face_load(mat_path, item)
            x = datashape_complete(face_roi)  # , video_path.replace('\n', ''))
            x = torch.stack(x, 0)

        elif self.dataset=='COHFACE':
            label_path = mat_path.with_name(mat_path.name.replace('.avi', '.txt'))
            label_f = open(label_path, 'r')
            label = label_f.readlines()
            label = ''.join(label)
            label = label.replace('[', '').replace(']', '').replace(',', ' ').split()
            label = list(map(float, label))
            face_roi = face_load(mat_path, item)
            x = datashape_complete(face_roi)  # , video_path.replace('\n', ''))
            x = torch.stack(x, 0)
        elif self.dataset=='PURE':
            label_path = mat_path.with_name(mat_path.name[5:].replace('.avi','.txt'))
            label_f = open(label_path, 'r')
            label=label_f.readline()
            label = label.replace('[', '').replace(']', '').replace(',', ' ').split()
            label = list(map(float, label))
            face_roi = face_load(mat_path, item)
            x = datashape_complete(face_roi)  # , video_path.replace('\n', ''))
            x = torch.stack(x, 0)
        else:
            label_path = mat_path.with_name(mat_path.name.replace('.avi', '.txt'))
            label_f = open(label_path, 'r')
            label = label_f.readline()
            label = label.replace('[', '').replace(']', '').replace(',', ' ').split()
            label = list(map(float, label))
            face_roi = face_load(mat_path, item)
            x = datashape_complete(face_roi)  # , video_path.replace('\n', ''))
            x = torch.stack(x, 0)

        label = heartpy.filter_signal(
            sample_rate=self.sample_rate,
            data=label,
            cutoff=[0.75, 2.75],
            filtertype='bandpass',
            order=1,
        )
        if self.dataset=='COHFACE':
            label=label[::2]

        try:
            _, m_label = heartpy.process(
                label,
                sample_rate=self.sample_rate,
                bpmmax=1e9,
                bpmmin=0,
                windowsize=8,
            )
            bpm = float(m_label.get('bpm', 75.0))
            if not np.isfinite(bpm):
                bpm = 75.0
        except Exception:
            bpm = 75.0

        # x shape: [T, H, W, C]
        label = torch.from_numpy(label.copy()).to(torch.float32).squeeze()

        # Only horizontal flip is recommended for face videos.
        x = self.random_video_flip(x, p=0.6, mode='horizontal')

        # Temporal resampling augmentation.
        x, label = self.resample_video_label_to_fixed_length(x, label, bpm)

        # Convert to model input shape: [C, T, H, W]
        x = x.permute(3, 0, 1, 2).contiguous()
        x = (x - torch.mean(x)) / torch.var(x)

        label_std = torch.std(label, unbiased=False)
        label = (label - torch.mean(label)) / (label_std + 1e-6)

        return x, label, item

    def random_video_flip(self, video: torch.Tensor, p: float = 0.5, mode: str = "horizontal") -> torch.Tensor:
        """
        Randomly flip a video tensor.

        Args:
            video: Tensor with shape [T, H, W, C].
            p: Probability of applying flip.
            mode: "horizontal", "vertical", or "both".

        Returns:
            Flipped or original video tensor with the same shape.
        """
        if not self.is_train:
            return video

        if video.ndim != 4:
            raise ValueError(f"Expected video shape [T, H, W, C], but got {video.shape}")

        if random.random() >= p:
            return video

        if mode == "horizontal":
            # For [T, H, W, C], horizontal flip means flipping width W.
            return torch.flip(video, dims=[2])

        if mode == "vertical":
            # Vertical flip is usually not recommended for face videos.
            return torch.flip(video, dims=[1])

        if mode == "both":
            # Keep this option for compatibility, but avoid frequent vertical flips.
            rand_value = random.random()
            if rand_value < 0.5:
                return torch.flip(video, dims=[2])
            return video

        raise ValueError("mode must be 'horizontal', 'vertical', or 'both'")

    def resample_video_label_to_fixed_length(
            self,
            video: torch.Tensor,
            label: torch.Tensor,
            hr: float,
    ):
        """
        Temporal resampling augmentation.

        Input video shape is [T, H, W, C]. The returned video has the same shape.
        """
        if not self.is_train:
            return video, label

        if video.ndim != 4:
            raise ValueError(f"Expected video shape [T, H, W, C], but got {video.shape}")
        video=video.to(dtype=torch.float32)
        D, H, W, C = video.shape
        data_aug = torch.zeros_like(video, dtype=torch.float32)
        labels_aug = torch.zeros((D,), dtype=torch.float32, device=label.device)

        # With 30% probability, keep the original sequence unchanged.
        if np.random.random() <= 0.3:
            return video, label

        if hr > 100:
            # Slow down high-HR samples by interleaving original frames and interpolated frames.
            # Choose a safe starting index for approximately half-length source frames.
            half_len = (D + 1) // 2
            max_start = max(0, D - half_len - 1)
            rand_start = random.randint(0, max_start) if max_start > 0 else 0

            even_indices = torch.arange(0, D, 2, device=label.device)
            odd_indices = even_indices + 1
            valid_odd_mask = odd_indices < D
            odd_indices = odd_indices[valid_odd_mask]

            src_even = torch.clamp(rand_start + even_indices // 2, max=D - 1)
            src_odd_1 = torch.clamp(rand_start + odd_indices // 2, max=D - 1)
            src_odd_2 = torch.clamp(src_odd_1 + 1, max=D - 1)

            # Use CPU indices for video if video is on CPU, which is the usual Dataset case.
            src_even_cpu = src_even.cpu()
            src_odd_1_cpu = src_odd_1.cpu()
            src_odd_2_cpu = src_odd_2.cpu()
            even_indices_cpu = even_indices.cpu()
            odd_indices_cpu = odd_indices.cpu()

            data_aug[even_indices_cpu] = video[src_even_cpu]
            labels_aug[even_indices] = label[src_even]

            data_aug[odd_indices_cpu] = (video[src_odd_1_cpu] + video[src_odd_2_cpu]) / 2.0
            labels_aug[odd_indices] = (label[src_odd_1] + label[src_odd_2]) / 2.0

        elif hr < 75:
            # Speed up low-HR samples by taking every other frame, then repeat to fixed length.
            sampled_video = video[::2]
            sampled_label = label[::2]
            sampled_len = sampled_video.shape[0]

            data_aug[:sampled_len] = sampled_video
            labels_aug[:sampled_len] = sampled_label

            remain_len = D - sampled_len
            if remain_len > 0:
                repeat_video = sampled_video[:remain_len]
                repeat_label = sampled_label[:remain_len]
                data_aug[sampled_len:] = repeat_video
                labels_aug[sampled_len:] = repeat_label

        else:
            data_aug = video
            labels_aug = label

        return data_aug, labels_aug


def split_subjects(root, test_ratio=0.25, seed=42):
    subject_dirs = sorted([p for p in Path(root).iterdir() if p.is_dir()])

    random.seed(seed)
    random.shuffle(subject_dirs)

    test_num = round(len(subject_dirs) * test_ratio)
    test_dirs = subject_dirs[:test_num]
    train_dirs = subject_dirs[test_num:]
    return train_dirs, test_dirs


def face_load(data_file,item):
    cap=cv.VideoCapture(str(data_file))
    face_roi=[]
    while (True):
        ret,frame=cap.read()
        if ret==False:
            break
        else:
            face_roi.append((frame))
    return face_roi

def datashape_complete(data):#,w_split,data_path):
    for j in range(len(data)):
        if j==0 and np.all(data[j]==0):
            for m in range(len(data)):
                if np.all(data[m]==0):
                    continue
                else:
                    data[j] = data[m + 1]
                    data[j] = cv.resize(data[j], (128, 128))
                    # data[j] = data_split(data[j], h_split, w_split, data_path)
                    break
        elif j!=0 and np.all(data[j]==0):
            data[j]=data[j-1]
        else:
            data[j] = cv.resize(data[j], (128, 128))
            # data[j] = data_split(data[j], h_split, w_split, data_path)
        data[j] = torch.as_tensor(data[j])
    return data

def build_dataloaders(
        root,
        Dataset_name,
        data_key,
        label_key,
        split_number=30,
        window_length=240,
        test_ratio=0.2,
        batch_size=8,
        seed=42,
        num_workers=0,
        persistent_workers=True,
        fixed_length=250,
        sample_rate=30,
        use_color_aug=True,
        use_srgb_linear=True,
        use_temporal_std=True,
        norm_clip_value=5.0,
):
    train_dirs, test_dirs = split_subjects(root, test_ratio, seed)

    train_dataset = Dataset(
        train_dirs,
        data_key,
        label_key,
        split_number,
        window_length,
        Dataset_name=Dataset_name,
        is_train=True,
        fixed_length=fixed_length,
        sample_rate=sample_rate
    )
    test_dataset = Dataset(
        test_dirs,
        data_key,
        label_key,
        split_number,
        window_length,
        Dataset_name=Dataset_name,
        is_train=False,
        fixed_length=fixed_length,
        sample_rate=sample_rate
    )

    # PyTorch requires persistent_workers=False when num_workers=0.
    effective_persistent_workers = persistent_workers and num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=effective_persistent_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=effective_persistent_workers,
    )

    return train_loader, test_loader


from pathlib import Path

def delete_target_files(root_dir):
    root = Path(root_dir)

    if not root.exists():
        raise FileNotFoundError(f"No File: {root}")

    # target_names = {"video.avi", "data.hdf5"}
    target_names = {"video.avi"}
    deleted_files = []

    for file_path in root.rglob("*"):
        if file_path.is_file() and file_path.name in target_names:
            try:
                file_path.unlink()
                deleted_files.append(file_path)
                print(f"Delete: {file_path}")
            except Exception as e:
                print(f"Delete Fail: {file_path}, Reason: {e}")

    print(f"\nDelete Finish，ALl {len(deleted_files)}  Files。")


# if __name__ == "__main__":
#     # root_dir = r"F:\LJC\DEAP\DEAP_unzip\face_video/"
#     # delete_target_files(root_dir)

