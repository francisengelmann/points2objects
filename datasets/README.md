# Datasets

### CoReNet Dataset

Download the corenet dataset:

```
for n in single pairs triplets; do  
  for s in train val test; do
    wget "https://storage.googleapis.com/gresearch/corenet/${n}.${s}.tar" \
      -O "data/raw/${n}.${s}.tar" 
    tar -xvf "data/raw/${n}.${s}.tar" -C data/ 
  done 
done
```

The corenet dataset uses the ```ShapeNetCore.v2.zip``` dataset, so we also need to download that one:

1. Create an account on https://shapenet.org/ (wait for it to be approved, this may take a while)
