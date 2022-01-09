import os
import math
import numpy as np
from absl import app
from absl import flags
from sklearn.cluster import KMeans
import pyviz3d.visualizer as viz
import datasets.utils as utils

FLAGS = flags.FLAGS
flags.DEFINE_string('sdf_path', '~/datasets/ShapeNetCore.v2.points_sdf', 'Prefix of path.')
flags.DEFINE_string('shapenet_path', '~/datasets/ShapeNetCore.v2', 'Prefix of path.')
flags.DEFINE_string('corenet_path', '~/datasets/corenet/data', 'Path to CoReNet dataset.')
flags.DEFINE_integer('k', 50, 'Number of clusters per model class')
flags.DEFINE_boolean('visualize', False, 'Visualize the cluster centers and the associated models.')


def load_sdf(models: dict) -> dict:
    sdf_from_name_from_class = {}
    for (model_class, mesh_name) in models:
        if model_class not in sdf_from_name_from_class:
            sdf_from_name_from_class[model_class] = {}
        sdf_filepath = os.path.join(FLAGS.sdf_path, model_class, mesh_name, 'models/model_normalized_sdf.npy')
        sdf_from_name_from_class[model_class][mesh_name] = np.load(os.path.expanduser(sdf_filepath))
    return sdf_from_name_from_class


def get_model_names(splits, pickle_file):
    if os.path.exists(pickle_file):
        model_names = utils.load_pkl(pickle_file)
    else:
        model_names = sorted(utils.corenet_models(FLAGS.corenet_path, splits=splits, datasets=['triplets', 'pairs']))
        utils.save_pkl(model_names, pickle_file)
    return model_names


def cluster():
    # Get the name of the models in the training set.
    train_models = get_model_names(['train'], 'training_models.pkl')
    train_val_test_models = get_model_names(['train', 'val', 'test'], 'train_val_test_models.pkl')

    # Load the SDFs of the training models.
    train_sdf_from_name_from_class = load_sdf(train_models)
    train_val_test_sdf_from_name_from_class = load_sdf(train_val_test_models)

    model_classes = train_sdf_from_name_from_class.keys()
    cluster_center_id_from_name_from_class = {}
    cluster_center_nearest_model_from_cluster_id = {}
    kmeans = KMeans(init='k-means++', n_clusters=FLAGS.k)

    # Cluster the training data and get cluster centers.
    for i, class_id in enumerate(model_classes):
        print(f'Cluster SDFs of class {class_id}...')
        sdf_from_name = train_sdf_from_name_from_class[class_id]
        model_data = np.concatenate([np.reshape(v, [1, -1]) for k, v in sdf_from_name.items()], axis=0)
        model_names = [k for k in sdf_from_name.keys()]
        kmeans.fit(model_data)

        print('Save per cluster information...')
        distances = kmeans.transform(model_data)
        nearest_model_id_per_cluster = np.argmin(distances, axis=0)
        for j in range(kmeans.cluster_centers_.shape[0]):
            cluster_id = j + (i * FLAGS.k)
            cluster_center = kmeans.cluster_centers_[j]
            nearest_model_name = model_names[nearest_model_id_per_cluster[j]]
            cluster_center_nearest_model_from_cluster_id[cluster_id] = (cluster_center, class_id, nearest_model_name)
    utils.save_pkl(cluster_center_nearest_model_from_cluster_id, 'dict_clusterCenter_class_nearestModel.pkl')

    # Predict cluster assignments of all models over all splits.
    for i, class_id in enumerate(model_classes):
        print('Assign cluster to the models over all splits...')
        sdf_from_name = train_val_test_sdf_from_name_from_class[class_id]
        model_data = np.concatenate([np.reshape(v, [1, -1]) for k, v in sdf_from_name.items()], axis=0)
        model_names = [k for k in sdf_from_name.keys()]
        labels = kmeans.predict(model_data)
        cluster_center_id_from_name_from_class[class_id] = {}
        for j in range(labels.shape[0]):
            model_name = model_names[j]
            cluster_id = labels[j] + (i * FLAGS.k)
            cluster_center_id_from_name_from_class[class_id][model_name] = cluster_id
    utils.save_pkl(cluster_center_id_from_name_from_class, 'dict_class_model_clusterId.pkl')


def visualize():
    cluster_center_id_from_name_from_class = utils.load_pkl('dict_class_model_clusterId.pkl')
    cluster_center_nearest_model_from_cluster_id = utils.load_pkl('dict_clusterCenter_class_nearestModel.pkl')

    model_class = '03001627'
    cluster_id_from_name = cluster_center_id_from_name_from_class[model_class]

    v = viz.Visualizer()
    for cluster_id in list(set([v for k, v in cluster_id_from_name.items()])):
        cluster_center_nearest_model = cluster_center_nearest_model_from_cluster_id[cluster_id]
        cluster_center_nearest_mesh_name = cluster_center_nearest_model[2]
        path_obj = os.path.join(os.path.expanduser(FLAGS.shapenet_path), model_class,
                                cluster_center_nearest_mesh_name, 'models/model_normalized.obj')
        v.add_mesh(f'Cluster_{cluster_id}', path=path_obj, rotation=[math.pi/2, 0.0, 0.0],
                   translation=[float(cluster_id % 50)/2.0, 0.0, 0.0], color=(np.random.rand(3) * 255).tolist())

    cluster_center_id_from_name = cluster_center_id_from_name_from_class[model_class]

    cluster_positions = {}
    for mesh_name, cluster_id in list(cluster_center_id_from_name.items())[0:50]:
        if cluster_id not in cluster_positions:
            cluster_positions[cluster_id] = 1
        else:
            cluster_positions[cluster_id] += 1
        path_obj = os.path.join(os.path.expanduser(FLAGS.shapenet_path), model_class,
                                mesh_name, 'models/model_normalized.obj')
        v.add_mesh(mesh_name, path=path_obj,
                   rotation=[math.pi/2, 0.0, 0.0], translation=[float(cluster_id % 50)/2.0, cluster_positions[cluster_id], 0.0],
                   color=(np.random.rand(3) * 255).tolist())
    v.save('cluster_vis')


def main(_):
    if FLAGS.visualize:
        visualize()
    else:
        cluster()


if __name__ == '__main__':
  app.run(main)
