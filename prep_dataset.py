import os
import time
import cv2
import lmdb
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm
def read_img_worker(path, key, compress_level):

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 2:
        h, w = img.shape
        c = 1
    else:
        h, w, c = img.shape
    _, img_byte = cv2.imencode('.png', img,
                               [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    return (key, img_byte, (h, w, c))

def scandir(dir_path, suffix=None, recursive=False, full_path=False):


    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)
def prepare_keys(folder_path, suffix='png'):

    img_path_list = sorted(
        list(scandir(folder_path, suffix=suffix, recursive=False)))
    keys = [img_path.split('.{}'.format(suffix))[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys
def make_lmdb_from_imgs(data_path,
                        lmdb_path,
                        img_path_list,
                        keys,
                        batch=5000,
                        compress_level=1,
                        multiprocessing_read=False,
                        n_thread=40,
                        map_size=None):

    assert len(img_path_list) == len(keys), (
        'img_path_list and keys should have the same length, '
        f'but got {len(img_path_list)} and {len(keys)}')
    print(f'Create lmdb for {data_path}, save to {lmdb_path}...')
    print(f'Total images: {len(img_path_list)}')
    if not lmdb_path.endswith('.lmdb'):
        raise ValueError("lmdb_path must end with '.lmdb'.")
    if osp.exists(lmdb_path):
        print(f'Folder {lmdb_path} already exists. Exit.')
        sys.exit(1)

    if multiprocessing_read:
        # read all the images to memory 
        dataset = {}  # use dict to keep the order
        shapes = {}
        print(f'Read images with multiprocessing, #thread: {n_thread} ...')
        pbar = tqdm(total=len(img_path_list), unit='image')

        def callback(arg):
            """get the image data and update pbar."""
            key, dataset[key], shapes[key] = arg
            pbar.update(1)
            pbar.set_description(f'Read {key}')

        pool = Pool(n_thread)
        for path, key in zip(img_path_list, keys):
            pool.apply_async(
                read_img_worker,
                args=(osp.join(data_path, path), key, compress_level),
                callback=callback)
        pool.close()
        pool.join()
        pbar.close()
        print(f'Finish reading {len(img_path_list)} images.')

    # create lmdb environment
    if map_size is None:
        # obtain data size for one image
        img = cv2.imread(
            osp.join(data_path, img_path_list[0]), cv2.IMREAD_UNCHANGED)
        _, img_byte = cv2.imencode(
            '.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
        data_size_per_img = img_byte.nbytes
        print('Data size per image is: ', data_size_per_img)
        data_size = data_size_per_img * len(img_path_list)
        map_size = data_size * 10

    env = lmdb.open(lmdb_path, map_size=map_size)

    # write data to lmdb
    pbar = tqdm(total=len(img_path_list), unit='chunk')
    txn = env.begin(write=True)
    txt_file = open(osp.join(lmdb_path, 'meta_info.txt'), 'w')
    for idx, (path, key) in enumerate(zip(img_path_list, keys)):
        pbar.update(1)
        pbar.set_description(f'Write {key}')
        key_byte = key.encode('ascii')
        if multiprocessing_read:
            img_byte = dataset[key]
            h, w, c = shapes[key]
        else:
            _, img_byte, img_shape = read_img_worker(
                osp.join(data_path, path), key, compress_level)
            h, w, c = img_shape

        txn.put(key_byte, img_byte)
        # write meta information
        txt_file.write(f'{key}.png ({h},{w},{c}) {compress_level}\n')
        if idx % batch == 0:
            txn.commit()
            txn = env.begin(write=True)
    pbar.close()
    txn.commit()
    env.close()
    txt_file.close()
    print('\nFinish writing lmdb.')

def create_lmdb_for_reds():

    folder_path = './datasets/REDS/train/train_sharp'
    lmdb_path = './datasets/REDS/train/train_sharp.lmdb'
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/REDS/train/train_blur_jpeg'
    lmdb_path = './datasets/REDS/train/train_blur_jpeg.lmdb'
    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
def make_val_300(folder, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    templates = '*9.*'
    cp_command = 'cp {} {}'.format(os.path.join(folder, templates), dst)
    os.system(cp_command)


def flatten_folders(folder):
    for vid in range(300):
        vidfolder_path = '{:03}'.format(vid)

        if not os.path.exists(os.path.join(folder, vidfolder_path)):
            continue

        print('working on .. {} .. {}'.format(folder, vid))
        for fid in range(100):
            src_filename = '{:08}'.format(fid)

            suffixes = ['.jpg', '.png']
            suffix = None

            for suf in suffixes:
                if os.path.exists(os.path.join(folder, vidfolder_path, src_filename+suf)):
                    suffix = suf
                    break
            assert suffix is not None


            src_filepath = os.path.join(folder, vidfolder_path, src_filename+suffix)
            dst_filepath = os.path.join(folder, '{}_{}{}'.format(vidfolder_path, src_filename, suffix))
            os.system('mv {} {}'.format(src_filepath, dst_filepath))
            time.sleep(0.001)
        os.system('rm -r {}'.format(os.path.join(folder, vidfolder_path)))


if __name__ == '__main__':
    flatten_folders('./datasets/REDS/train/train_blur_jpeg')
    flatten_folders('./datasets/REDS/train/train_sharp')

    create_lmdb_for_reds()

