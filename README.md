# Occluded Primitives

### Datasets

```/cns/li-d/home/engelmann/occluded_primitives/data/```

Scripts to generate the datasets are in ```occluded_primitives/data```.
##### Kostas' Synthetic Chair Dataset
To create the tfrecord files for Kostas chair dataset run:
```data/xm_kostas.sh```.  
Modify ```data/xm_kostas.py``` to generate train or test split.

##### Train CoReNet pairs:
```
occluded_primitives/train_multi_objects/train.py
--alsologtostderr
--stderrthreshold=info
--logdir='/usr/local/google/home/engelmann/occluded_primitives/logs/n8/'
--tfrecords_dir='/usr/local/google/home/engelmann/occluded_primitives/data_corenet/triplets'
--shapenet_dir='/usr/local/google/home/engelmann/occluded_primitives/shapenet'
--xm_runlocal
--number_hourglasses=1
--num_overfitting_samples=3
--num_classes=14
--max_num_objects=2
--soft_shape_labels=True
--collision_weight=1.0
--run_graph=False
--batch_size=1
--learning_rate=0.0001
--debug=True
--train
```

##### Validate CoReNet triplets
```
occluded_primitives/train_multi_objects/train.py
--alsologtostderr
--stderrthreshold=info
--logdir='/cns/li-d/home/engelmann/occluded_primitives/multi_objects/logs/18543882/35-pose_pc_pc_weight=10.0,projected_pose_pc_pc_weight=0.1,shapes_weight=0.01,sizes_3d_weight=100.0,soft_shape_labels=True,soft_shape_labels_a=32'
--tfrecords_dir='/usr/local/google/home/engelmann/occluded_primitives/data_corenet/triplets'
--max_num_objects=3
--xm_runlocal
--run_graph=False
--debug=False
--soft_shape_labels=True
--part_id=-2
--val
--eval_only=True
--local_plot_3d=False
--qualitative=True
```


Predicts the oriented 3D bounding box of a single object in the scene.

```shell script
google_xmanager launch experimental/giotto/occluded_primitives/single_object/xm_launch.py -- \
--xm_resource_pool=perception \
--xm_resource_alloc=group:perception/giotto3d \
--noxm_monitor_on_launch
```# points2objects
