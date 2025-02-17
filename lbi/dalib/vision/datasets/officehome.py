import os
from typing import Optional
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class OfficeHome(ImageList):
    """`OfficeHome <http://hemanthdv.org/OfficeHome-Dataset/>`_ Dataset.

    Parameters:
        - **root** (str): Root directory of dataset
        - **task** (str): The task (domain) to create dataset. Choices include ``'Ar'``: Art, \
            ``'Cl'``: Clipart, ``'Pr'``: Product and ``'Rw'``: Real_World.
        - **download** (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        - **transform** (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, ``transforms.RandomCrop``.
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            Art/
                Alarm_Clock/*.jpg
                ...
            Clipart/
            Product/
            Real_World/
            image_list/
                Art.txt
                Clipart.txt
                Product.txt
                Real_World.txt
    """
    download_list = [
        ("image_list", "image_list.zip",
         "https://cloud.tsinghua.edu.cn/f/ee615d5ad5e146278a80/?dl=1"),
        ("Art", "Art.tgz",
         "https://cloud.tsinghua.edu.cn/f/81a4f30c7e894298b435/?dl=1"),
        ("Clipart", "Clipart.tgz",
         "https://cloud.tsinghua.edu.cn/f/d4ad15137c734917aa5c/?dl=1"),
        ("Product", "Product.tgz",
         "https://cloud.tsinghua.edu.cn/f/a6b643999c574184bbcd/?dl=1"),
        ("Real_World", "Real_World.tgz",
         "https://cloud.tsinghua.edu.cn/f/60ca8452bcf743408245/?dl=1")
    ]
    image_list = {
        "Ar": "image_list/Art.txt",
        "Cl": "image_list/Clipart.txt",
        "Pr": "image_list/Product.txt",
        "Rw": "image_list/Real_World.txt",
        "Ar_train": "image_list/Art_train.txt",
        "Ar_test": "image_list/Art_test.txt",
        "Ar_val": "image_list/Art_val.txt",
        "Cl_train": "image_list/Clipart_train.txt",
        "Cl_test": "image_list/Clipart_test.txt",
        "Cl_val": "image_list/Clipart_val.txt",
        "Pr_train": "image_list/Product_train.txt",
        "Pr_test": "image_list/Product_test.txt",
        "Pr_val": "image_list/Product_val.txt",
        "Rw_train": "image_list/Real_World_train.txt",
        "Rw_test": "image_list/Real_World_test.txt",
        "Rw_val": "image_list/Real_World_val.txt",
        "ArCl_train": "image_list/Art_Clipart_train.txt",
        "ClAr_train": "image_list/Art_Clipart_train.txt",
        "ArPr_train": "image_list/Art_Product_train.txt",
        "PrAr_train": "image_list/Art_Product_train.txt",
        "ArRw_train": "image_list/Art_RealWorld_train.txt",
        "RwAr_train": "image_list/Art_RealWorld_train.txt",
        "ClPr_train": "image_list/Clipart_Product_train.txt",
        "PrCl_train": "image_list/Clipart_Product_train.txt",
        "ClRw_train": "image_list/Clipart_RealWorld_train.txt",
        "RwCl_train": "image_list/Clipart_RealWorld_train.txt",
        "PrRw_train": "image_list/Product_RealWorld_train.txt",
        "RwPr_train": "image_list/Product_RealWorld_train.txt"
    }
    CLASSES = [
        'Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet',
        'Shelf', 'Toys', 'Sink', 'Laptop', 'Kettle', 'Folder', 'Keyboard',
        'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch', 'Bike',
        'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet',
        'Mouse', 'Pen', 'Monitor', 'Mop', 'Sneakers', 'Notebook', 'Backpack',
        'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio', 'Fan',
        'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker',
        'Eraser', 'Bucket', 'Chair', 'Calendar', 'Calculator', 'Flowers',
        'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
        'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator',
        'Marker'
    ]

    def __init__(self,
                 root: str,
                 task: str,
                 download: Optional[bool] = False,
                 **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        if task == 'Ar_train':
            domain_idx = 0
        elif task == 'Cl_train':
            domain_idx = 1
        elif task == 'Pr_train':
            domain_idx = 2
        elif task == 'Rw_train':
            domain_idx = 3
        else:
            domain_idx = -1

        if download:
            list(
                map(lambda args: download_data(root, *args),
                    self.download_list))
        else:
            list(
                map(lambda file_name, _: check_exits(root, file_name),
                    self.download_list))

        super(OfficeHome, self).__init__(root,
                                         OfficeHome.CLASSES,
                                         data_list_file=data_list_file,
                                         domain_idx=domain_idx,
                                         **kwargs)
