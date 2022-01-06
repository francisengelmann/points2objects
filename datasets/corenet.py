import os
import json

dataset_dir = '/home/fengelmann/datasets/corenet/data'

for dataset in ['triplets', 'pairs']:
  for split in ['test', 'val', 'train']:
    dataset_path = os.path.join(dataset_dir, f'{dataset}.{split}')
    dataset_json_file = os.path.join(dataset_path, 'dataset.json')
    with open(dataset_json_file, 'r') as f:
      json_file = json.load(f)
      classes = json_file['classes']
      classes_names = [c['human_readable'] for c in classes]
      classes_ids = [c['id'] for c in classes]
      print(split, dataset, classes_names, classes_ids)

for i in range(len(classes_names)):
  class_name = classes_names[i]
  class_id = classes_ids[i]
  print('| ' + class_id + ' | ' + class_name + ' \t |')
