# Setup

### 1. Change Directory to maskbit

Set up the environment and data in maskbit dir which contain the codes.

```shell
cd ./maskbit
```

---

### 2. Prepare the Environment

```shell
conda create -n maskbit python=3.11
conda activate maskbit
pip install -r requirements.txt
pip install -e .
```

---

### 3. Prepare the Data

```shell
cd ./data
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
cd ..
```

---

### 4. Transform Data to torchvision Format

Transform the tar file into torchvision datasets format:

```python
from torchvision.datasets import ImageNet

ds = ImageNet("./data")
ds_val = ImageNet("/data", split='val')
```

---

### 5. Prepare Dataset Shards

```shell
mkdir ./shards
python scripts/create_sharded_dataset.py --data="./data" --shards="./shards"
```
or if it is too slow for interactive job, submit it as a batched job
```bash
sbatch shard_data.sh
```

If the resulting data is not in `maskbit/shards`, modify the path in `run_slurm.sh` to be the path for data shards.

---

### 6. Modify Paths and Work Directory

Change the work directory in `activateEnvironment.sh`.

---

### 7. Run Training Script

Change the number of processes if necessary.

```shell
bash run_slurm.sh
```



