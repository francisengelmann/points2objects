import os
import io
import json

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
import PIL.Image


FLAGS = flags.FLAGS
flags.DEFINE_string('corenet_path', '~/datasets/corenet/data', 'Path to corenet dataset.')
flags.DEFINE_string('cluster_path', 'dict_class_model_clusterId.pkl', 'Path to cluster file.')
flags.DEFINE_string('tfrecords_path', '~/datasets/corenet_tfrecords/', 'Path to the .tfrecord files.')
flags.DEFINE_string('split', 'train', '{train, val, test}')
flags.DEFINE_string('dataset', 'triplets', '{triplets, pairs}')
flags.DEFINE_integer('tfrecord_id', 0, '')
flags.DEFINE_integer('part', 0, '')
flags.DEFINE_integer('num_tfrecords', 100, '')
flags.DEFINE_boolean('francis', False, 'Debugging mode.')
flags.DEFINE_boolean('debug', False, 'Debug mode.')


def print_dataset_statistics() -> None:
    for dataset in ['triplets', 'pairs']:
        for split in ['test', 'val', 'train']:
            dataset_path = os.path.join(os.path.expanduser(FLAGS.corenet_path), f'{dataset}.{split}')
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


def get_files_from_datasets_and_splits(dataset: str, split: str) -> [str]:
    json_filepath = os.path.join(os.path.expanduser(FLAGS.corenet_path), f'{dataset}.{split}', 'dataset.json')
    with open(json_filepath) as f:
        json_file = json.load(f)
        return json_file['files']


def create_tfrecord_writer():
    writer_name = f'{FLAGS.split}-{FLAGS.tfrecord_id:05d}-of-{FLAGS.num_tfrecords}.tfrecord'
    tfrecords_filepath = os.path.join(os.path.expanduser(FLAGS.tfrecords_path), writer_name)
    os.makedirs(os.path.dirname(tfrecords_filepath), exist_ok=True)
    return tf.io.TFRecordWriter(tfrecords_filepath)


def get_tfrecord_file_ids(files, tfrecord_id, num_tfrecords):
    return np.array_split(np.arange(0, len(files)), num_tfrecords)[tfrecord_id]


def process_corenet_scene(scene_npz):
    scene = dict()
    scene['filename'] = scene_npz["scene_filename"].item().decode()
    scene['mesh_labels'] = scene_npz["mesh_labels"].tolist()
    scene['mesh_filenames'] = scene_npz["mesh_filenames"].tolist()
    scene['mesh_visible_fractions'] = np.array(scene_npz["mesh_visible_fractions"], np.float32)
    scene['mesh_object_to_world_transforms'] = np.array(scene_npz["mesh_object_to_world_transforms"], np.float32)
    scene['view_transform'] = np.array(scene_npz["view_transform"], np.float32)
    scene['camera_transform'] = np.array(scene_npz["camera_transform"], np.float32)
    scene['opengl_image'] = np.array(PIL.Image.open(io.BytesIO(scene_npz["opengl_image"])), np.uint8)
    scene['pbrt_image'] = np.array(PIL.Image.open(io.BytesIO(scene_npz["pbrt_image"])), np.uint8)
    # todo ...

    return scene


def main(_) -> None:
    files = get_files_from_datasets_and_splits(dataset='triplets', split=FLAGS.split)
    writer = create_tfrecord_writer()
    tfrecord_file_ids = get_tfrecord_file_ids(files, FLAGS.tfrecord_id, FLAGS.num_tfrecords)

    for file_id in tfrecord_file_ids:
        scene_filepath = os.path.join(os.path.expanduser(FLAGS.corenet_path),
                                      f'{FLAGS.dataset}.{FLAGS.split}', files[file_id])
        scene = np.load(scene_filepath)
        sample = process_corenet_scene(scene)
        writer.write(sample.SerializeToString())
    writer.close()


if __name__ == '__main__':
    app.run(main)
