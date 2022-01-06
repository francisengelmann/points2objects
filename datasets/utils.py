import os
import json
import numpy as np
import trimesh
from skimage import measure

os.environ['PYOPENGL_PLATFORM'] = 'egl'  # rendering without gui
from mesh_to_sdf import mesh_to_voxels


def sum_up(array: np.array):
    """Helper function to convert trimesh.Scenes into a single mesh.
    array.sum() would also work but results in OOM in some cases.
    :param array: meshes of scene
    :return: single mesh
    """
    if len(array) == 1:
        return array[0]
    if len(array) == 2:
        return array.sum()
    else:
        length = len(array)
        return sum_up(array[0:length // 2]) + sum_up(array[length // 2:])


def load_mesh(path: str):
    """Load mesh from given path.
    :param path: mesh path
    :return: the mesh
    """
    return trimesh.load(path)


def normalize_mesh(mesh):
    """Translate and rescale (non-uniform) the mesh vertices to the [-1,1]^3 cube.
    :param mesh: The mesh.
    :return: The normalized mesh.
    """
    if isinstance(mesh, trimesh.Scene):
        mesh_array = mesh.dump()
        mesh = sum_up(mesh_array)
    vertices = mesh.vertices - mesh.bounding_box.centroid
    vertices *= 2 / mesh.bounding_box.extents
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def sample_points_from_mesh(mesh, num_samples=512):
    """Sample points from the surface of the mesh.
    :param mesh: The mesh from which to sample points.
    :param num_samples: The number of sampled points.
    :return: Sampled points.
    """
    points, _ = trimesh.sample.sample_surface(mesh, num_samples)
    return points


def create_sdf_from_mesh(mesh):
    """Computes the signed distance function (SDF) from the given mesh.
    :param mesh: The input mesh.
    :return: The output SDF voxel grid.
    """
    return mesh_to_voxels(mesh, voxel_resolution=30,
                          surface_point_method='scan',
                          sign_method='depth',
                          scan_count=20,
                          scan_resolution=400,
                          pad=True,
                          check_result=False)


def save(data: np.array, path: str) -> None:
    """Save data to specified path.
    :param data: The numpy data.
    :param path: The path where to save the file.
    :return:
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    np.save(path, data)


def screenshot(sdf, path):
    """Visualize the specified SDF.
    :param sdf: The sdf of the mesh to visualize.
    :param path: The path where to save the visualization.
    """
    os.environ['DISPLAY'] = ':1'
    voxels = sdf
    path_png = path
    vertices, faces, normals, _ = measure.marching_cubes(voxels, level=0.04)
    mesh_sdf = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    if not os.path.exists(os.path.dirname(path_png)):
        os.makedirs(os.path.dirname(path_png))
    scene = trimesh.scene.Scene()
    mesh_sdf.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 4.0, [1.0, 0, 0]))
    scene.add_geometry(mesh_sdf)
    png_image = scene.save_image(resolution=[256, 256], visible=True)
    with open(os.path.expanduser(path_png + '_sdf.png'), 'wb') as f:
        f.write(png_image)
        f.close()


def corenet_models(dataset_dir, datasets=None, splits=None):
    """ Collect the class and name of all shapenet models in the corenet datasets.
    :param dataset_dir: path to corenet dataset
    :param datasets: list of sub-datasets of corenet
    :param splits: list of corenet splits
    :return: set of tuples {(mesh_type, mesh_name)_i}
    """

    print('Collecting CoReNet models from ShapeNet...')
    if datasets is None:
        datasets = ['triplets', 'pairs']
    if splits is None:
        splits = ['test', 'val', 'train']
    models_set = set()
    dataset_dir = os.path.expanduser(dataset_dir)
    for dataset in datasets:
        for split in splits:
            print(f'\t - {dataset} {split}')
            dataset_path = os.path.join(dataset_dir, f'{dataset}.{split}')
            dataset_json_file = os.path.join(dataset_path, 'dataset.json')
            with open(dataset_json_file, 'r') as f:
                json_file = json.load(f)
                files = json_file['files']
                for file in files:
                    scene_path = os.path.join(dataset_path, file)
                    scene = np.load(scene_path)
                    for label, filename in zip(scene["mesh_labels"].tolist(), scene["mesh_filenames"].tolist()):
                        models_set.add((label, filename))
    return models_set
