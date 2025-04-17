import os
import os.path
import numpy as np
from PIL import Image
import torch
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
from typing import Callable, Optional, Tuple, cast
from torchvision.datasets.utils import check_integrity, verify_str_arg
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from sklearn.preprocessing import OneHotEncoder
from augment.randaugment import RandomAugment

mean, std = {}, {}

mean['stl10'] = [x / 255 for x in [112.4, 109.1, 98.6]]
std['stl10'] = [x / 255 for x in [68.4, 66.6, 68.5]]


class MY_STL10(VisionDataset):
    base_folder = "stl10_binary"
    url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    filename = "stl10_binary.tar.gz"
    tgz_md5 = "91f7769df0f17e558f3565bffb0c7dfb"
    class_names_file = "class_names.txt"
    folds_list_file = "fold_indices.txt"
    train_list = [
        ["train_X.bin", "918c2871b30a85fa023e0c44e0bee87f"],
        ["train_y.bin", "5a34089d4802c674881badbb80307741"],
        ["unlabeled_X.bin", "5242ba1fed5e4be9e1e742405eb56ca4"],
    ]

    test_list = [["test_X.bin", "7f263ba9f9e0b06b93213547f721ac82"], ["test_y.bin", "36f9794fa4beb8a2c72628de14fa638e"]]
    splits = ("train", "train+unlabeled", "unlabeled", "test")

    def __init__(
            self,
            root: str,
            args,
            folds: Optional[int] = None,
            split: str = "train+unlabeled",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False, rate_partial=0.3,
            labeled_idx=None, unlabeled_idx=None, lb_dataset=False
    ) -> None:
        super(MY_STL10, self).__init__(root, transform=transform,
                                       target_transform=target_transform)
        self.split = verify_str_arg(split, "split", self.splits)
        self.folds = self._verify_folds(folds)

        self.args = args
        self.labeled_idx = labeled_idx
        self.unlabeled_idx = unlabeled_idx
        self.lb_dataset = lb_dataset

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.labels: Optional[np.ndarray]
        if self.split == "train":
            self.data, self.labels = self.__loadfile(self.train_list[0][0], self.train_list[1][0])
            self.labels = cast(np.ndarray, self.labels)
            self.__load_folds(folds)

        elif self.split == "train+unlabeled":
            self.data, self.labels = self.__loadfile(self.train_list[0][0], self.train_list[1][0])
            self.labels = cast(np.ndarray, self.labels)
            self.__load_folds(folds)
            unlabeled_data, _ = self.__loadfile(self.train_list[2][0])
            self.data = np.concatenate((self.data, unlabeled_data))
            self.labels = np.concatenate((self.labels, np.asarray([-1] * unlabeled_data.shape[0])))

        elif self.split == "unlabeled":
            self.data, _ = self.__loadfile(self.train_list[2][0])
            self.labels = np.asarray([-1] * self.data.shape[0])
        else:  # self.split == 'test':
            self.data, self.labels = self.__loadfile(self.test_list[0][0], self.test_list[1][0])

        self.targets = torch.tensor(self.labels)

        crop_size = 96
        crop_ratio = 0.875
        self.weak_transform = Compose([
            Resize(crop_size),
            RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)),
                       padding_mode='reflect'),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean['stl10'], std['stl10'])
        ])

        self.strong_transform = Compose([
            Resize(crop_size),
            RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)),
                       padding_mode='reflect'),
            RandomHorizontalFlip(),
            RandomAugment(3, 5),
            ToTensor(),
            Normalize(mean['stl10'], std['stl10'])
        ])

        self.rate_partial = rate_partial

        # 如果给出了lb_idx，则构造lb数据集
        if labeled_idx is not None and lb_dataset is True:
            # lb_targets = np.array(self.targets)[labeled_idx]
            # lb_targets_one_hot = F.one_hot(torch.tensor(lb_targets, dtype=torch.int64), num_classes=args.num_classes).type(torch.float32)
            # average_partial_lb_targets, _ = partialize(lb_targets_one_hot, args.partial_rate)
            self.data = self.data[labeled_idx]
            self.targets = self.targets[labeled_idx]
            self.partial_labels = self.generate_partial_labels()
            args.logger.info(f"Use {args.num_labels} labeled partial samples for training")
        # 如果给出了ulb_idx，则构造ulb数据集，ulb数据集把所有标签当做候选标记
        elif unlabeled_idx is not None and lb_dataset is False:
            unlabeled_idx = unlabeled_idx + np.arange(5000, len(unlabeled_data) + 5000).tolist()
            self.data = self.data[unlabeled_idx]
            # average_partial_lb_targets = torch.ones((len(unlabeled_idx), args.num_classes)) / args.num_classes
            self.partial_labels = torch.ones((len(unlabeled_idx), args.num_classes))
            self.targets = self.targets[unlabeled_idx]
            args.logger.info(f"Use {len(unlabeled_idx)} unlabeled partial samples for training")
        # 常规偏标记设定
        else:
            self.partial_labels = self.generate_partial_labels()
            args.logger.info(f"Use {len(self.data)} labeled partial samples for training")

    def _verify_folds(self, folds: Optional[int]) -> Optional[int]:
        if folds is None:
            return folds
        elif isinstance(folds, int):
            if folds in range(10):
                return folds
            msg = "Value for argument folds should be in the range [0, 10), but got {}."
            raise ValueError(msg.format(folds))
        else:
            msg = "Expected type None or int for argument folds, but got type {}."
            raise ValueError(msg.format(type(folds)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, partial_label = self.data[index], self.targets[index], self.partial_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.weak_transform is not None:
            img_weak = self.weak_transform(img)
            img_strong = self.strong_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.labeled_idx is not None or not hasattr(self.args, 'num_labels'):
            return img_weak, target, partial_label, index
        else:
            return img_weak, img_strong, target, partial_label, index

    def __len__(self) -> int:
        return self.data.shape[0]

    def __loadfile(self, data_file: str, labels_file: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        labels = None
        if labels_file:
            path_to_labels = os.path.join(self.root, self.base_folder, labels_file)
            with open(path_to_labels, "rb") as f:
                labels = np.fromfile(f, dtype=np.uint8) - 1  # 0-based

        path_to_data = os.path.join(self.root, self.base_folder, data_file)
        with open(path_to_data, "rb") as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))

        return images, labels

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
        self._check_integrity()

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

    def __load_folds(self, folds: Optional[int]) -> None:
        # loads one of the folds if specified
        if folds is None:
            return
        path_to_folds = os.path.join(self.root, self.base_folder, self.folds_list_file)
        with open(path_to_folds) as f:
            str_idx = f.read().splitlines()[folds]
            list_idx = np.fromstring(str_idx, dtype=np.int64, sep=" ")
            self.data = self.data[list_idx, :, :, :]
            if self.labels is not None:
                self.labels = self.labels[list_idx]

    def generate_partial_labels(self):
        def binarize_class(y):
            label = y.reshape(len(y), -1)
            enc = OneHotEncoder(categories='auto')
            enc.fit(label)
            label = enc.transform(label).toarray().astype(np.float32)
            label = torch.from_numpy(label)
            return label

            new_y = binarize_class(train_labels.clone())
            n, c = new_y.shape[0], new_y.shape[1]
            avgC = 0

        new_y = binarize_class(self.targets)
        n = len(self.targets)
        c = max(self.targets) + 1
        avgC = 0
        partial_rate = self.rate_partial
        print(partial_rate)
        for i in range(n):
            row = new_y[i, :]
            row[np.where(np.random.binomial(1, partial_rate, c) == 1)] = 1
            while torch.sum(row) == 1:
                row[np.random.randint(0, c)] = 1
            avgC += torch.sum(row)
            new_y[i] = row

        avgC = avgC / n
        print("Finish Generating Candidate Label Sets:{}!\n".format(avgC))
        new_y = new_y.cpu().numpy()
        return new_y
