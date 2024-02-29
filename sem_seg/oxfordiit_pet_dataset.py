from typing import Any, Callable, Optional, Sequence, Tuple, Union
from torchvision.datasets import VisionDataset
import numpy as np
from glob import glob
import os
import os.path
import pathlib
from PIL import Image
from torchvision import datapoints

class OxfordIIITPetDataset(VisionDataset):
    """`Oxford-IIIT Pet Dataset   <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_.
    Modified version from 
    https://pytorch.org/vision/main/generated/torchvision.datasets.OxfordIIITPet.html?highlight=oxford#torchvision.datasets.OxfordIIITPet
    Args:
        root (string): Root directory of the dataset.
        file_list (list): List of the jpg files to be used in the dataset as input. 
        transforms (callable, optional): A function/transforms that takes in an image and a label and returns the transformed versions of both.
        download (bool, optional): If True, downloads the dataset from the internet and puts it into
            ``root/oxford-iiit-pet``. If dataset is already downloaded, it is not downloaded again.
    """

    _RESOURCES = (
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
    )

    def __init__(
        self,
        root: str,
        file_list: Sequence[str],
        transforms: Optional[Callable] = None,
        download: bool = False,
    ):
        
        super().__init__(root, transforms=transforms)
        self._base_folder = pathlib.Path(self.root) 
        self._images_folder = self._base_folder / "images"
        self._anns_folder = self._base_folder / "annotations"
        self._segs_folder = self._anns_folder / "trimaps"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        
        self._images = [self._images_folder / f"{image}" for image in file_list]
        self._segs = [self._segs_folder/ f"{image.replace('.jpg','.png')}" for image in file_list]

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = Image.open(self._images[idx]).convert("RGB")
        target = Image.open(self._segs[idx])

        if self.transforms:
            image = datapoints.Image(image)
            target = datapoints.Mask(target)
            image, target = self.transforms(image, target)

        return image, (target - 1).long()

    def _check_exists(self) -> bool:
        for folder in (self._images_folder, self._anns_folder):
            if not (os.path.exists(folder) and os.path.isdir(folder)):
                return False
        else:
            return True

    def _download(self) -> None:
        if self._check_exists():
            return

        for url, md5 in self._RESOURCES:
            download_and_extract_archive(url, download_root=str(self._base_folder), md5=md5)
def get_datasets(root:str,
                 splits: Union[Sequence[float],float],
                 transforms: Optional[Callable]=None,
                 download: bool=False) -> Tuple[type(OxfordIIITPetDataset)]:
    """
    Create datasets based on splits.
    inputs:
        root: (str) root dir of dataset. Something like 'your_path/oxford-iiit-pet'
        splits: (list[float] | float). If float indicates the split for training set and the split for validation set is (1 - splits).
                If list the first two are the train and validation split and the test split is 1 - sum(splits).
        transforms: (torch list) Sequence of torchvision transfomations to apply to the datasets.
        download: (bool) Download the dataset if not found in root.
    """
    if not isinstance(splits,list):
        splits = [splits]
    assert len(splits) < 3, f'Provide a max of two splits. Third is inferred if needed.'
    splits = np.array(splits)
    assert splits.sum() < 1, f'The total split cannot be greater than 1'
    splits = np.append(splits,(1 - splits.sum()))
    #create list of all files available
    ls = glob(os.path.join(root,'images','*.jpg'))
    ls = np.array([os.path.basename(l) for l in ls])
    indx = np.arange(len(ls))
    #shuffle done in place
    np.random.shuffle(indx)
    #shuffle the list of files
    ls = ls[indx]
    #create the indices for each split
    file_splits = [0]
    file_splits.extend(np.cumsum(splits)*len(ls))
    file_splits = np.array(file_splits).astype(int)
    #get datesets
    datasets = []
    for i in range(1,len(file_splits)):
        file_list = ls[file_splits[i-1]:file_splits[i]]
        datasets.append(OxfordIIITPetDataset(root,file_list,transforms,download))
    return datasets