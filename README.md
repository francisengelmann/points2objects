# Points2Objects

Points2Objects is an approach for reconstructing multiple objects from a single image.
You can find more information in the following paper:
[From Points to Multi-Object 3D Reconstruction](https://openaccess.thecvf.com/content/CVPR2021/html/Engelmann_From_Points_to_Multi-Object_3D_Reconstruction_CVPR_2021_paper.html).
If you find the code or the paper useful, please consider citing our work:
```bibtex
@inproceedings{Engelmann21CVPR,
  author = {Engelmann, Francis and Rematas, Konstantinos and Leibe, Bastian and Ferrari, Vittorio},
  title = {{From Points to Multi-Object 3D Reconstruction}},
  booktitle = {{IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}},
  year = {2021}
}
```

## Preprocessing

#### 1) Download CoReNet dataset

```bash
for n in single pairs triplets; do  
  for s in train val test; do
    wget "https://storage.googleapis.com/gresearch/corenet/${n}.${s}.tar" \
      -O "data/raw/${n}.${s}.tar" 
    tar -xvf "data/raw/${n}.${s}.tar" -C data/ 
  done 
done
```

#### 2) Download ShapeNet dataset

The CoReNet dataset itself does not contain any polygon meshes.
Instead, it relies on the ```ShapeNetCore.v2``` dataset.
```bash
wget https://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2.zip --no-check-certificate
unzip ShapeNetCore.v2.zip
```

The 14 ShapeNet classes used in the CoReNet datasets (pairs and triplets) are as follows:

| Class ID | Class Name  |
|----------|-------------|
| 02818832 | bed 	       |
| 02876657 | bottle 	    |
| 02880940 | bowl 	      |
| 02958343 | car 	       |
| 03001627 | chair 	     |
| 03211117 | display 	   |
| 03467517 | guitar 	    |
| 03636649 | lamp 	      |
| 03790512 | motorcycle  |
| 03797390 | mug 	       |
| 03928116 | piano 	     |
| 03938244 | pillow 	    |
| 04256520 | sofa 	      |
| 04379243 | table 	     |

#### 3) Generate point clouds and SDFs
For each mesh, we sample points from its surface to obtain a point cloud,
and we compute a signed distance function (SDF) representation.
This step will take a while.
You can keep track of the progress and verify that all files have been successfully processed using
`ls ~/datasets/ShapeNetCore.v2.points_sdf/* -l | wc -l` which should show `27881`.

```bash
cd datasets
python preprocess.py --shapenet_path='~/datasets/ShapeNetCore.v2' --corenet_path='~/datasets/corenet/data' --output_path='~/datasets/ShapeNetCore.v2.points_sdf'
```

#### 4) Shape clustering
This step generates two pickle files (`dict_class_model_clusterId.pkl`, `dict_clusterCenter_class_nearestModel.pkl`) used by the model during training and evaluation.
```bash
cd datasets
python cluster.py --shapenet_path='~/datasets/ShapeNetCore.v2' --corenet_path='~/datasets/corenet/data' --sdf_path='~/datasets/ShapeNetCore.v2.points_sdf'
```

#### 5) Generating tfrecord files
WIP