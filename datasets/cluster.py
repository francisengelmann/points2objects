import os
import pickle
import numpy as np
from absl import app
from absl import flags
from sklearn.cluster import KMeans
import utils

FLAGS = flags.FLAGS
flags.DEFINE_string('sdf_path', '~/datasets/ShapeNetCore.v2.points_sdf', 'Prefix of path.')
flags.DEFINE_string('corenet_path', '~/datasets/corenet/data', 'Path to CoReNet dataset.')
flags.DEFINE_integer('k', 50, 'Number of clusters per model class')


def main(_):
    # Get the name of the models in the training set
    training_models_path_debug = 'training_models.pkl'
    if os.path.exists(training_models_path_debug):
        train_models = utils.load_pkl(training_models_path_debug)
    else:
        train_models = sorted(utils.corenet_models(FLAGS.corenet_path, splits=['train'], datasets=['triplets', 'pairs']))
        utils.save_pkl(train_models, 'training_models.pkl')

    # Load the SDFs of the training models
    sdf_from_name_from_class = {}
    for (model_class, mesh_name) in train_models:
        if model_class not in sdf_from_name_from_class:
            sdf_from_name_from_class[model_class] = {}
        sdf_filepath = os.path.join(FLAGS.sdf_path, model_class, mesh_name, 'models/model_normalized_sdf.npy')
        sdf_from_name_from_class[model_class][mesh_name] = np.load(os.path.expanduser(sdf_filepath))

    print('Number of models per class:')
    for k in sdf_from_name_from_class.keys():
        print(k, len(sdf_from_name_from_class[k]))

    dict_class_model_cluster_id = {}
    dict_cluster_center_class_nearest_model = {}

    kmeans = KMeans(init='k-means++', n_clusters=FLAGS.k)
    for i, class_id in enumerate(sdf_from_name_from_class.keys()):
        print(f'Cluster SDFs of class {class_id}...')
        sdf_from_name = sdf_from_name_from_class[class_id]
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
            dict_cluster_center_class_nearest_model[cluster_id] = (cluster_center, class_id, nearest_model_name)

        labels = kmeans.predict(model_data)
        dict_class_model_cluster_id[class_id] = {}
        for j in range(labels.shape[0]):
            model_name = model_names[j]
            cluster_id = labels[j] + (i * FLAGS.k)
            dict_class_model_cluster_id[class_id][model_name] = cluster_id

    with open(os.path.join('dict_class_model_clusterId.pkl'), 'wb') as f:
        pickle.dump(dict_class_model_cluster_id, f)

    with open(os.path.join('dict_clusterCenter_class_nearestModel.pkl'), 'wb') as f:
        pickle.dump(dict_cluster_center_class_nearest_model, f)


if __name__ == '__main__':
  app.run(main)
