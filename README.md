1. cd into the maskbit dir. All the codes are currently in maskbit dir
```shell
cd ./maskbit
```
2. prepare the env
```shell
conda create -n maskbit python=3.11
conda activate maskbit
pip install -r requirements.txt
pip install -e .
```
3. prepare the data
```shell
cd ./data
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
cd ..
```
    
Transform the tar file into torchivision datasets format
```python
from torchvision.datasets import ImageNet

ds = ImageNet("./data")
ds_val = ImageNet("/data", split='val')
```

Prepare the dataset shards
```shell
mkdir ./shards
python scripts/create_sharded_dataset.py --data="./data" --shards="./shards"
```
or
```bash
sbatch shard_data.sh
```

If the resulting data is not in ``maskbit/shards``, modify the path in ``run_slurm.sh`` to be the path for data shards


6. run training script. change the num processes if necessary
```shell
bash run_slurm.sh
```

    

