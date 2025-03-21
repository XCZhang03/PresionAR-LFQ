1. cd into the maskbit dir 
```shell
cd ./maskbit
```
2. prepare the env
```shell
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
```

Prepare the dataset shards
```shell
python scripts/create_sharded_dataset.py
```

If the resulting data is not in ``maskbit/shards``, modify the path in ``config.dataset`` where config is the config file in the train script

5. Prepare wandb
```shell
wandb init
```
api key ``78319f33ffd79b3480286266fd8ba1f9c5bc3dab``

6. run training script. change the num processes if necessary
```shell
bash train.sh
```

    

