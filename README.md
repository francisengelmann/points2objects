# Points2Objects


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
For each mesh, we sample points from its surface to obtain a point cloud, and we compute a signed distance function SDF.
This steo will take a while to process.

```bash
cd datasets
python preprocess.py --shapenet_path='~/datasets/ShapeNetCore.v2' --corenet_path='~/datasets/corenet/data' --output_path='~/datasets/ShapeNetCore.v2.points_sdf'
```
You can veryify that all files have been succesfully processed using
`ls ~/datasets/ShapeNetCore.v2.points_sdf/* -l | wc -l` which should show `27881`.
