import os
import json
import glob
from collections import defaultdict

#展示大图数量和小图数量
LABELS_DIR = "../../data/raw_data/labels"
OUTPUT_DIR = "../../data/classifier_dataset_hsv"
CLASS_NAMES = ['nn49','nn60','zld105','sn29','lk314','nn47','jd17','sd30','hd16','jng20839','nn43','nn42','sn23','b73','sz2','zd57','xd18','zd53','zd61','zd59','zd51','sd29','xzd1','zh301','nn55']

large_counts = defaultdict(int)
for json_path in glob.glob(os.path.join(LABELS_DIR, "*.json")):
    with open(json_path, 'r', encoding='utf-8') as f:
        labels = set(s['label'] for s in json.load(f).get('shapes', []) if s['label'] in CLASS_NAMES)
        for label in labels:
            large_counts[label] += 1

print(f"{'类别':<15}{'大图数量':<15}{'真实小图':<15}{'增强小图':<15}{'总小图'}")
for c in CLASS_NAMES:
    all_imgs = glob.glob(os.path.join(OUTPUT_DIR, "*", c, "*.jpg"))
    aug_imgs = [img for img in all_imgs if "_aug" in img]
    print(f"{c:<15}{large_counts[c]:<15}{len(all_imgs)-len(aug_imgs):<15}{len(aug_imgs):<15}{len(all_imgs)}")