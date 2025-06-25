import os
import shutil

base_dir = 'data3'
images_dir = os.path.join(base_dir, 'images')
labels_dir = os.path.join(base_dir, 'labels')

splits = {'train': 'train_files.txt', 'valid': 'val_files.txt'}

for split, split_file in splits.items():
    print(f"Processing {split} split...")
    os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, split, 'labels'), exist_ok=True)
    
    with open(os.path.join(base_dir, split_file), 'r') as f:
        img_files = [line.strip() for line in f.readlines()]
    
    for idx, img_file in enumerate(img_files):
        if idx % 100 == 0:
            print(f"{split}: {idx}/{len(img_files)} files processed")
        # Copy image
        src_img = os.path.join(images_dir, img_file)
        dst_img = os.path.join(base_dir, split, 'images', img_file)
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)
        else:
            print(f"Image not found: {src_img}")
        
        # Copy label
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label = os.path.join(labels_dir, label_file)
        dst_label = os.path.join(base_dir, split, 'labels', label_file)
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)
        else:
            print(f"Label not found: {src_label}")
