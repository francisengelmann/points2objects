# Points2Objects


# Preprocessing

### Download CoReNet Dataset

```
for n in single pairs triplets; do  
  for s in train val test; do
    wget "https://storage.googleapis.com/gresearch/corenet/${n}.${s}.tar" \
      -O "data/raw/${n}.${s}.tar" 
    tar -xvf "data/raw/${n}.${s}.tar" -C data/ 
  done 
done
```

### Download ShapeNet Dataset

The CoReNet dataset itself does not contain any polygon meshes.
Instead, it relies on the ```ShapeNetCore.v2``` dataset.
```
wget https://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2.zip --no-check-certificate
unzip ShapeNetCore.v2.zip
```

The 14 ShapeNet classes used in the CoReNet datasets (pairs and triplets) are as follows:

| Class ID | Class Name |
|----------|------------|
| 02818832 | bed 	 |
| 02876657 | bottle 	 |
| 02880940 | bowl 	 |
| 02958343 | car 	 |
| 03001627 | chair 	 |
| 03211117 | display 	 |
| 03467517 | guitar 	 |
| 03636649 | lamp 	 |
| 03790512 | motorcycle 	 |
| 03797390 | mug 	 |
| 03928116 | piano 	 |
| 03938244 | pillow 	 |
| 04256520 | sofa 	 |
| 04379243 | table 	 |

## Creating the point clouds and SDFs
