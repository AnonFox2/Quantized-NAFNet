
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from image_functions import apply_transforms, crop_matching_patches
from utils import imfrombytes, img2tensor, padding
from utils import LmdbBackend
from os import path as osp

def paired_paths_from_lmdb(folders, keys):

    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    if not (input_folder.endswith('.lmdb') and gt_folder.endswith('.lmdb')):
        raise ValueError(
            f'{input_key} folder and {gt_key} folder should both in lmdb '
            f'formats. But received {input_key}: {input_folder}; '
            f'{gt_key}: {gt_folder}')
    # ensure that the two meta_info files are the same
    with open(osp.join(input_folder, 'meta_info.txt')) as fin:
        input_lmdb_keys = [line.split('.')[0] for line in fin]
    with open(osp.join(gt_folder, 'meta_info.txt')) as fin:
        gt_lmdb_keys = [line.split('.')[0] for line in fin]
    if set(input_lmdb_keys) != set(gt_lmdb_keys):
        raise ValueError(
            f'Keys in {input_key}_folder and {gt_key}_folder are different.')
    else:
        paths = []
        for lmdb_key in sorted(input_lmdb_keys):
            paths.append(
                dict([(f'{input_key}_path', lmdb_key),
                      (f'{gt_key}_path', lmdb_key)]))
        return paths

class Dataloader(data.Dataset):
    def __init__(self, gt,lq,im_size,train=False):
        super(Dataloader, self).__init__()

        self.is_train=train
        self.im_size=im_size
        self.file_client = None


        self.gt_folder, self.lq_folder = gt, lq

        self.filename_tmpl = '{}'

        self.paths = paired_paths_from_lmdb(
            [self.lq_folder, self.gt_folder], ['lq', 'gt'])

    def __getitem__(self, index):

        gt_path = self.paths[index]['gt_path']

        lm_files=LmdbBackend(db_paths= [self.lq_folder, self.gt_folder],client_keys=['lq', 'gt'])
        img_bytes = lm_files.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)


        lq_path = self.paths[index]['lq_path']
        img_bytes = lm_files.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))


        # augmentation for training
        if self.is_train:
            gt_size = self.im_size

            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = crop_matching_patches(img_gt, img_lq, gt_size, gt_path)
            # flip, rotation
            img_gt, img_lq = apply_transforms([img_gt, img_lq])

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
