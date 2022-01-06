import os
from absl import app
from absl import flags
import utils
import numpy as np
import multiprocessing

FLAGS = flags.FLAGS
flags.DEFINE_string('shapenet_path', '~/datasets/ShapeNetCore.v2', '')
flags.DEFINE_string('corenet_path', '~/datasets/corenet/data', '')
flags.DEFINE_string('model_class', '03790512', '')
flags.DEFINE_string('mesh_name', 'fb5d82c273bdad39359c7139c3be4ec1', '')
flags.DEFINE_string('output_path', '~/datasets/ShapeNetCore.v2.points_sdf', '')
flags.DEFINE_boolean('process_all', False, 'Ignore specified model and process all.')
flags.DEFINE_boolean('show_sdf', True, 'Visualize SDF of specified model for debugging.')


def prepare_paths(dataset_path, model_class, mesh_name, output_path):
    """Assemble the necessary paths from the given dataset path and models.
    :param dataset_path:
    :param model_class:
    :param mesh_name:
    :param output_path:
    :return:
    """
    mesh_path = os.path.join(dataset_path, model_class, mesh_name, 'models/model_normalized.obj')
    points_path = os.path.join(output_path, model_class, mesh_name, 'models/model_normalized_points.npy')
    sdf_path = os.path.join(output_path, model_class, mesh_name, 'models/model_normalized_sdf.npy')
    return os.path.expanduser(mesh_path), os.path.expanduser(points_path), os.path.expanduser(sdf_path)


def process_mesh(mesh_path, points_path, sdf_path):
    points_exists = os.path.exists(points_path)
    sdf_exists = os.path.exists(sdf_path)
    if points_exists and sdf_exists:
        return
    mesh = utils.load_mesh(mesh_path)
    mesh_normalized = utils.normalize_mesh(mesh)
    if not points_exists:
        mesh_points = utils.sample_points_from_mesh(mesh_normalized)
        utils.save(mesh_points, points_path)
    else:
        print('Exists points: ', points_path)
    if not sdf_exists:
        mesh_sdf = utils.create_sdf_from_mesh(mesh_normalized)
        utils.save(mesh_sdf, sdf_path)
    else:
        print('Exists sdf: ', sdf_path)
    print('Done: ', mesh_path)


def process_mesh_wrapper(inputs):
    model_class, mesh_name = inputs
    mesh_path, points_path, sdf_path = prepare_paths(FLAGS.shapenet_path, model_class, mesh_name, FLAGS.output_path)
    process_mesh(mesh_path, points_path, sdf_path)


def main(_):
    if FLAGS.process_all:
        print(f'Processing all meshes...')
        models_per_class = utils.corenet_models(FLAGS.corenet_path)
        # with open('models_per_class.pickle', 'wb') as handle:
        #     pickle.dump(models_per_class, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # return
        print(f"Number of shapenet models: {len(models_per_class)}")  # 27729 / 27840
        # with open('models_per_class.pickle', 'rb') as handle:
        #     print('Loading pickle')
        #     models_per_class = pickle.load(handle)
        #     for inputs in sorted(models_per_class):
        #         print(inputs)
        #         process_mesh_wrapper(inputs)
        pool_obj = multiprocessing.Pool(24)
        pool_obj.map(process_mesh_wrapper, models_per_class)
    if FLAGS.show_sdf:
        print(f'Visualize compute SDF for debugging {FLAGS.model_class} {FLAGS.mesh_name}...')
        _, _, sdf_path = prepare_paths(FLAGS.shapenet_path,
                                       FLAGS.model_class,
                                       FLAGS.mesh_name,
                                       FLAGS.output_path)
        sdf = np.load(sdf_path)
        utils.screenshot(sdf, os.path.join(FLAGS.output_path, f'{FLAGS.model_class}_{FLAGS.mesh_name}'))
    else:
        print(f'Processing mesh {FLAGS.model_class} {FLAGS.mesh_name}...')
        mesh_path, points_path, sdf_path = prepare_paths(FLAGS.shapenet_path,
                                                         FLAGS.model_class,
                                                         FLAGS.mesh_name,
                                                         FLAGS.output_path)
        process_mesh(mesh_path, points_path, sdf_path)


if __name__ == '__main__':
    app.run(main)
