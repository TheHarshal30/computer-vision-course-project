import os
import glob
import random
import shutil

imgs = []
for ext in ["*.jpg", "*.png", "*.jpeg"]:
    imgs.extend(glob.glob(f'data/merged_dataset/images/{ext}'))

random.seed(42)
random.shuffle(imgs)
val_imgs = set(imgs[:int(len(imgs)*0.15)])

os.makedirs('data/merged_dataset/images/train', exist_ok=True)
os.makedirs('data/merged_dataset/images/val', exist_ok=True)
os.makedirs('data/merged_dataset/labels/train', exist_ok=True)
os.makedirs('data/merged_dataset/labels/val', exist_ok=True)

for i in imgs:
    is_val = i in val_imgs
    sub = 'val' if is_val else 'train'
    
    # move image
    dst_img = f'data/merged_dataset/images/{sub}/{os.path.basename(i)}'
    shutil.move(i, dst_img)
    
    # map to old label location (currently everything is in labels/train)
    old_lbl = i.replace('/images/', '/labels/train/')
    old_lbl = os.path.splitext(old_lbl)[0] + '.txt'
    
    # move label
    if os.path.exists(old_lbl):
        dst_lbl = f'data/merged_dataset/labels/{sub}/{os.path.basename(old_lbl)}'
        shutil.move(old_lbl, dst_lbl)

print(f"Split done: {len(imgs)-len(val_imgs)} train, {len(val_imgs)} val.")
