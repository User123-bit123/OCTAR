# encoding: utf-8
import glob
import os.path as osp
import re

from .bases import BaseImageDataset


class OccludedREID(BaseImageDataset):
    """
    Occluded-REID.

    The dataset is usually used only for testing:
      - query: occluded_body_images
      - gallery: whole_body_images

    To keep the existing CLIP-ReID / TransReID training and model construction
    pipeline working, we expose a placeholder train split built from both query
    and gallery images with relabeled IDs.
    """

    dataset_dir = "Occluded_REID"

    def __init__(self, root="", verbose=True, pid_begin=0, **kwargs):
        super(OccludedREID, self).__init__()
        self.dataset_dir = self._resolve_dataset_dir(root)
        self.query_dir = osp.join(self.dataset_dir, "occluded_body_images")
        self.gallery_dir = osp.join(self.dataset_dir, "whole_body_images")
        self.pid_begin = pid_begin

        self._check_before_run()

        query = self._process_dir(self.query_dir, camid=0, relabel=False)
        gallery = self._process_dir(self.gallery_dir, camid=1, relabel=False)
        train = self._process_dir(self.query_dir, camid=0, relabel=True) + \
            self._process_dir(self.gallery_dir, camid=1, relabel=True)

        if verbose:
            print("=> Occluded-REID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _resolve_dataset_dir(self, root):
        root = str(root)
        candidates = [
            osp.join(root, "Occluded_REID"),
            osp.join(root, "OccludedREID"),
            osp.join(root, "Occluded-ReID"),
            root,
        ]
        for cand in candidates:
            if osp.exists(osp.join(cand, "occluded_body_images")) and osp.exists(osp.join(cand, "whole_body_images")):
                return cand
            if osp.exists(osp.join(cand, "query")) and osp.exists(osp.join(cand, "gallery")):
                return cand
        return osp.join(root, self.dataset_dir)

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, camid, relabel=False):
        img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            img_paths.extend(glob.glob(osp.join(dir_path, ext)))
            img_paths.extend(glob.glob(osp.join(dir_path, "*", ext)))
        img_paths = sorted(set(img_paths))
        if not img_paths:
            raise RuntimeError("No images found in '{}'".format(dir_path))

        pid_container = set()
        parsed = []
        for img_path in img_paths:
            pid = self._parse_pid(img_path)
            pid_container.add(pid)
            parsed.append((img_path, pid))

        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
        dataset = []
        for img_path, pid in parsed:
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset

    @staticmethod
    def _parse_pid(img_path):
        parent = osp.basename(osp.dirname(img_path))
        if re.fullmatch(r"\d+", parent):
            return int(parent)

        name = osp.basename(img_path)
        match = re.match(r"(\d+)[_-]", name)
        if match:
            return int(match.group(1))

        match = re.match(r"(\d+)", name)
        if match:
            return int(match.group(1))

        raise RuntimeError("Cannot parse pid from '{}'".format(img_path))
