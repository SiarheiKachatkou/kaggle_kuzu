import mmcv
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

DATASET_PATH=Path('data/kuzushiji-recognition_dbg')
MAX_TRAIN_IMGS=10

def iter_bboxes(labels):
    if not labels:
        return
    labels = labels.split()
    n = len(labels)
    assert n % 5 == 0
    for i in range(0, n, 5):
        ch, x, y, w, h = labels[i:i + 5]
        yield int(x), int(y), int(w), int(h), ch


def prepare_train():
    df = pd.read_csv(DATASET_PATH/'train.csv', keep_default_na=False)
    img_dir = DATASET_PATH/'train_images'

    unicode_translation = pd.read_csv(DATASET_PATH/'unicode_translation.csv')
    unicode2class = dict(
        zip(unicode_translation['Unicode'], unicode_translation.index.values))

    # Add images to COCO
    images = []
    for img_id, row in tqdm(df.iterrows()):
        filename = row['image_id'] + '.jpg'
        img = Image.open(img_dir / filename)
        image = {
            'filename': filename,
            'width': img.width,
            'height': img.height,
        }
        bboxes = []
        labels = []
        for x, y, w, h, ch in iter_bboxes(row['labels']):
            if ch in unicode2class:
                bboxes.append([x, y, x + w, y + h])
                # labels.append(1)
                labels.append(unicode2class[ch] + 1)

        if labels:
            image['ann'] = {
                'bboxes': np.array(bboxes).astype(np.float32).reshape(-1, 4),
                'labels': np.array(labels).astype(np.int64).reshape(-1),
                'bboxes_ignore': np.array([], dtype=np.float32).reshape(-1, 4),
                'labels_ignore': np.array([], dtype=np.int64).reshape(-1, )
            }
            images.append(image)

    import random
    random.shuffle(images)
    if MAX_TRAIN_IMGS is not None:
        images=images[:MAX_TRAIN_IMGS]
    mmcv.dump([im for im in images if im['filename'].startswith('umgy')], DATASET_PATH/'dval.pkl')
    mmcv.dump([im for im in images if not im['filename'].startswith('umgy')], DATASET_PATH/'dtrain.pkl')
    mmcv.dump(images, DATASET_PATH/'dtrainval.pkl')


def prepare_test():
    df = pd.read_csv(DATASET_PATH/'sample_submission.csv', keep_default_na=False)
    img_dir = Path(DATASET_PATH/'test_images')

    images = []
    for img_id, row in tqdm(df.iterrows()):
        filename = row['image_id'] + '.jpg'
        img = Image.open(img_dir / filename)
        images.append({
            'filename': filename,
            'width': img.width,
            'height': img.height,
            'ann': {
                'bboxes': np.array([], dtype=np.float32).reshape(-1, 4),
                'labels': np.array([], dtype=np.int64).reshape(-1, ),
                'bboxes_ignore': np.array([], dtype=np.float32).reshape(-1, 4),
                'labels_ignore': np.array([], dtype=np.int64).reshape(-1, )
            }
        })
    mmcv.dump(images, DATASET_PATH/'dtest.pkl')


if __name__ == "__main__":
    prepare_train()
    prepare_test()