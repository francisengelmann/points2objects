import os
import json
import numpy as np
import trimesh
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from mesh_to_sdf import mesh_to_voxels


def load_mesh(path):
  return trimesh.load(path)


def normalize_mesh(mesh):
  if isinstance(mesh, trimesh.Scene):
    mesh = mesh.dump().sum()
  vertices = mesh.vertices - mesh.bounding_box.centroid
  vertices *= 2 / (mesh.bounding_box.extents)
  return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def sample_points_from_mesh(mesh, num_samples=512):
  points, _ = trimesh.sample.sample_surface(mesh, num_samples)
  return points


def create_sdf_from_mesh(mesh):
  return mesh_to_voxels(mesh, voxel_resolution=30,
                 surface_point_method='scan',
                 sign_method='depth',
                 scan_count=20,
                 scan_resolution=400,
                 pad=True,
                 check_result=False)


def save(data, path):
  if not os.path.exists(os.path.dirname(path)):
    os.makedirs(os.path.dirname(path))
  np.save(path, data)


def screenshot(sdf, path):
  from skimage import measure
  voxels = sdf
  path_png = path
  vertices, faces, normals, _ = measure.marching_cubes(voxels, level=0.04)
  mesh_sdf = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
  mesh_sdf.show()

  if not os.path.exists(os.path.dirname(path_png)):
    os.makedirs(os.path.dirname(path_png))

  scene = trimesh.scene.Scene()
  mesh_sdf.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 4.0, [1.0, 0, 0]))
  scene.add_geometry(mesh_sdf)
  png_image = scene.save_image(resolution=[256, 256], visible=True)
  if not os.path.exists(os.path.dirname(path_png)):
    os.makedirs(os.path.dirname(path_png))
  with open(path_png + '_sdf.png', 'wb') as f:
    f.write(png_image)
  f.close()


def corenet_models(dataset_dir, datasets=['triplets', 'pairs'], splits=['test', 'val', 'train']):
    models_set = set()
    for dataset in datasets:
        for split in splits:
            print(dataset, split)
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
