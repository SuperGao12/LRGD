from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
from ..utils.data import BaseImageDataset


class regdb_ir(BaseImageDataset):
    """
    regdb_ir
    train in market1501 type data
    test in orignal regdb data
    """
    dataset_dir = 'ir_modify/'

    def __init__(self, root, trial= 0,verbose=True, **kwargs):
        # verbose：一个布尔值，用于控制是否打印加载数据集的信息，默认为 True
        super(regdb_ir, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, str(trial)+'/'+'bounding_box_train')

        self.query_dir = osp.join(self.dataset_dir, str(trial)+'/'+'query') 
        self.gallery_dir = osp.join(self.dataset_dir, str(trial)+'/'+'bounding_box_test')


        self._check_before_run()
        # 调用 self._check_before_run() 方法检查这些路径是否存在

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        # 调用 self._process_dir 方法分别处理训练集、查询集和图库集目录，得到相应的数据集列表

        if verbose:
            print("=> regdb_ir loaded",trial)
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.bmp'))
        # 获取图像路径：使用 glob.glob 函数获取指定目录下所有 .bmp 格式的图像文件路径。
        pattern = re.compile(r'([-\d]+)_c(\d)')
        # 使用 re.compile 编译一个正则表达式 r'([-\d]+)_c(\d)'，用于从图像文件名中提取行人 ID 和摄像头 ID。

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        # 遍历所有图像路径，提取行人 ID 并添加到 pid_container 集合中，忽略行人 ID 为 -1 的图像。

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
            # 重新标记行人 ID：如果 relabel 为 True，则创建一个字典 pid2label，将原始的行人 ID 映射到连续的整数标签。
            # 处理图像路径：再次遍历所有图像路径，提取行人 ID 和摄像头 ID，进行必要的检查和调整（如将摄像头 ID 索引从 1 调整为 0），
            # 如果需要则重新标记行人 ID，最后将图像路径、行人 ID 和摄像头 ID 组成的元组添加到 dataset 列表中。
            # 返回数据集：返回处理好的数据集列表。

        return dataset
