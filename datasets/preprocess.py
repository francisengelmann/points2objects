import os
from absl import app
from absl import flags
import utils
import multiprocessing

FLAGS = flags.FLAGS
flags.DEFINE_string('shapenet_path', '/home/fengelmann/datasets/ShapeNetCore.v2', '')
flags.DEFINE_string('corenet_path', '/home/fengelmann/datasets/corenet/data', '')
flags.DEFINE_string('model_class', '02818832', '')
flags.DEFINE_string('mesh_name', '2d1a2be896054548997e2c877588ae24', '')
flags.DEFINE_string('output_path', '/home/fengelmann/datasets/ShapeNetCore.v2.points_sdf', '')
flags.DEFINE_boolean('process_all', True, 'Ignore specified model and process all.')


def prepare_pathes(dataset_path, model_class, mesh_name, output_path):
    mesh_path = os.path.join(dataset_path, model_class, mesh_name, 'models/model_normalized.obj')
    points_path = os.path.join(output_path, model_class, mesh_name, 'models/model_normalized_points.npy')
    sdf_path = os.path.join(output_path, model_class, mesh_name, 'models/model_normalized_sdf.npy')
    return os.path.expanduser(mesh_path), os.path.expanduser(points_path), os.path.expanduser(sdf_path)


def process_mesh(mesh_path, points_path, sdf_path):
    mesh = utils.load_mesh(mesh_path)
    mesh_normalized = utils.normalize_mesh(mesh)
    if not os.path.exists(points_path):
        mesh_points = utils.sample_points_from_mesh(mesh_normalized)
        utils.save(mesh_points, points_path)
    else:
        print('Exists: ', points_path)
    if not os.path.exists(sdf_path):
        mesh_sdf = utils.create_sdf_from_mesh(mesh_normalized)
        utils.save(mesh_sdf, sdf_path)
    else:
        print('Exists: ', sdf_path)
    print('Done: ', sdf_path)


def process_mesh_wrapper(inputs):
    model_class, mesh_name = inputs
    mesh_path, points_path, sdf_path = prepare_pathes(FLAGS.shapenet_path, model_class, mesh_name, FLAGS.output_path)
    process_mesh(mesh_path, points_path, sdf_path)

def main(_):
    if FLAGS.process_all:
        models_per_class = utils.corenet_models(FLAGS.corenet_path)
        pool_obj = multiprocessing.Pool(24)
        pool_obj.map(process_mesh_wrapper, models_per_class)
    else:
        mesh_path, points_path, sdf_path = prepare_pathes(FLAGS.shapenet_path,
                                                          FLAGS.model_class,
                                                          FLAGS.mesh_name,
                                                          FLAGS.output_path)
        process_mesh(mesh_path, points_path, sdf_path)


if __name__ == '__main__':
    app.run(main)
