import os
import multiprocessing
import numpy as np
from absl import app
from absl import flags
import utils

FLAGS = flags.FLAGS
flags.DEFINE_string('shapenet_path', '~/datasets/ShapeNetCore.v2', 'Path to shapenet dataset.')
flags.DEFINE_string('corenet_path', '~/datasets/corenet/data', 'Path to corenet dataset.')
flags.DEFINE_string('output_path', '~/datasets/ShapeNetCore.v2.points_sdf', 'Path to generated points and SDFs.')
flags.DEFINE_boolean('process_all', True, 'Ignore specified model and process all.')
flags.DEFINE_string('model_class', '03790512', 'Class of specific model (for debugging).')
flags.DEFINE_string('mesh_name', 'fb5d82c273bdad39359c7139c3be4ec1', 'Name of specific mesh (for debugging).')
flags.DEFINE_boolean('show_sdf', False, 'Visualize SDF of specified model (for debugging).')
flags.DEFINE_integer('num_processes', 24, 'Number of parallel processes.')


def prepare_paths(dataset_path: str, model_class: str, mesh_name: str, output_path: str) -> (str, str, str):
    """Assemble the necessary paths from the given dataset path and models.
    :param dataset_path: Path to the ShapeNet dataset.
    :param model_class: Class of the specified model.
    :param mesh_name: Name of the specified mesh.
    :param output_path: Where to write the generated points and SDFs.
    :return: Tuple of mesh path, points path and output path.
    """
    mesh_path = os.path.join(dataset_path, model_class, mesh_name, 'models/model_normalized.obj')
    points_path = os.path.join(output_path, model_class, mesh_name, 'models/model_normalized_points.npy')
    sdf_path = os.path.join(output_path, model_class, mesh_name, 'models/model_normalized_sdf.npy')
    return os.path.expanduser(mesh_path), os.path.expanduser(points_path), os.path.expanduser(sdf_path)


def process_mesh(mesh_path: str, points_path: str, sdf_path: str) -> None:
    """ Samples points from the surface of the specified mesh and generates an SDF.
    :param mesh_path: Path to the ShapeNet mesh.
    :param points_path: Path to save the sampled surface points.
    :param sdf_path: Path to save the generated SDF.
    """
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


def process_mesh_wrapper(inputs: (str, str)) -> None:
    """Helper function for multiprocessing. Prepares the paths and processes the mesh (compute surface points and sdf).
    :param inputs: Tuple of the model class and mesh name.
    """
    model_class, mesh_name = inputs
    mesh_path, points_path, sdf_path = prepare_paths(FLAGS.shapenet_path, model_class, mesh_name, FLAGS.output_path)
    process_mesh(mesh_path, points_path, sdf_path)


def main(_):
    if FLAGS.process_all:
        print(f'Processing all meshes...')
        models_per_class = utils.corenet_models(FLAGS.corenet_path)
        print(f"Number of shapenet models: {len(models_per_class)}")
        pool_obj = multiprocessing.Pool(FLAGS.num_processes)
        pool_obj.map(process_mesh_wrapper, models_per_class)
    elif FLAGS.show_sdf:
        print(f'Visualize compute SDF for debugging {FLAGS.model_class} {FLAGS.mesh_name}...')
        _, _, sdf_path = prepare_paths(FLAGS.shapenet_path, FLAGS.model_class, FLAGS.mesh_name, FLAGS.output_path)
        sdf = np.load(sdf_path)
        utils.screenshot(sdf, os.path.join(FLAGS.output_path, f'{FLAGS.model_class}_{FLAGS.mesh_name}'))
    else:
        print(f'Processing mesh {FLAGS.model_class} {FLAGS.mesh_name}...')
        mesh_path, points_path, sdf_path = prepare_paths(FLAGS.shapenet_path, FLAGS.model_class, FLAGS.mesh_name,
                                                         FLAGS.output_path)
        process_mesh(mesh_path, points_path, sdf_path)


if __name__ == '__main__':
    app.run(main)
