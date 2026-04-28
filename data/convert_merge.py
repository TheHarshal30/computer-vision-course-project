import os
import glob
import shutil

# Unified class IDs
CLS_HELMET        = 0
CLS_NO_HELMET     = 1
CLS_MOTORCYCLE    = 2
CLS_LICENSE_PLATE = 3

src_map = {
    "data/tmp/license_plate": {
        0: CLS_LICENSE_PLATE  # 'Number_Plate'
    },
    "data/tmp/traffic_violation": {
        0: CLS_MOTORCYCLE,    # '3-person detection on 2-wheeler' -> motorcycle
        1: CLS_HELMET,        # 'With Helmet'
        2: CLS_NO_HELMET      # 'Without Helmet'
    }
}

out_dir = "data/merged_dataset"
os.makedirs(f"{out_dir}/images", exist_ok=True)
os.makedirs(f"{out_dir}/labels/train", exist_ok=True)

def obb_to_aabb(coords):
    # coords: [x1, y1, x2, y2, x3, y3, x4, y4]
    xs = coords[0::2]
    ys = coords[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    
    xc = (xmin + xmax) / 2.0
    yc = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    
    # clamp values slightly to prevent floating point issues going > 1.0
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    
    return xc, yc, w, h

for src_dir, mapping in src_map.items():
    print(f"Processing {src_dir} ...")
    
    # Process train, valid, test splits inside the source
    for split in ["train", "valid", "test"]:
        labels_dir = os.path.join(src_dir, split, "labels")
        images_dir = os.path.join(src_dir, split, "images")
        
        if not os.path.isdir(labels_dir):
            continue
            
        for txt_file in glob.glob(f"{labels_dir}/*.txt"):
            base = os.path.splitext(os.path.basename(txt_file))[0]
            
            # Find image
            img_ext = None
            for ext in [".jpg", ".png", ".jpeg"]:
                if os.path.exists(os.path.join(images_dir, base + ext)):
                    img_ext = ext
                    break
                    
            if not img_ext:
                continue
                
            img_path = os.path.join(images_dir, base + img_ext)
            out_img_path = os.path.join(out_dir, "images", f"{os.path.basename(src_dir)}_{base}{img_ext}")
            out_txt_path = os.path.join(out_dir, "labels", "train", f"{os.path.basename(src_dir)}_{base}.txt")
            
            # Copy image
            if not os.path.exists(out_img_path):
                shutil.copy(img_path, out_img_path)
            
            # Convert label
            with open(txt_file, "r") as f_in, open(out_txt_path, "w") as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if not parts: continue
                    
                    src_id = int(parts[0])
                    if src_id not in mapping:
                        continue
                        
                    new_id = mapping[src_id]
                    
                    # Check if standard YOLO (4 coords) or OBB (8 coords)
                    coords = [float(p) for p in parts[1:]]
                    if len(coords) >= 8:
                        xc, yc, w, h = obb_to_aabb(coords[:8])
                    elif len(coords) == 4:
                        xc, yc, w, h = coords
                    else:
                        continue # Malformed
                        
                    f_out.write(f"{new_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

print("Conversion complete!")
