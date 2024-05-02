# How to train your custom model for OBS Detect plugin

OBS Detect is based on the [EdgeYOLO](https://github.com/LSH9832/edgeyolo) work.
They provide a model training script that works with just setting some parameters.

You need to get a dataset first. The supported dataset formats are mentiond in the [EdgeYOLO](https://github.com/LSH9832/edgeyolo?tab=readme-ov-file#train) readme: COCO, VOC, YOLO, and DOTA.

In this example we will use a COCO dataset from Roboflow. You can get the dataset from [here](https://public.roboflow.com/object-detection/aquarium/2).

The dataset is in the COCO format, so we can use it directly with the EdgeYOLO training script.

## Step 1: Unpack the dataset

Unzip the dataset to a folder. The dataset should have the following structure:

```plaintext
dataset_folder/
    train/
        _annotations.coco.json
        image1.jpg
        image2.jpg
        ...
    valid/
        _annotations.coco.json
        image1.jpg
        image2.jpg
        ...
    test/
        _annotations.coco.json
        image1.jpg
        image2.jpg
        ...
```

## Step 2: Install the required packages

You need to install the required packages to train the model. You can install them using the following command:

```bash
pip install -r requirements.txt
```

## Step 3: Setup the parameters of the training script

You need to set the parameters of the training script.
Make a copy of the configuration file and set the parameters according to your needs.
For example for my case, I will set the following parameters in the `params/train/train_coco_aquarium.yaml` file:

```yaml
# models & weights------------------------------------------------------------------------------------------------------
model_cfg: "params/model/edgeyolo_tiny.yaml"         # model structure config file
weights: "output/train/edgeyolo_tiny_coco_aquarium/last.pth"  # contains model_cfg, set null or a no-exist filename if not use it
use_cfg: false                                       # force using model_cfg instead of cfg in weights to build model

# output----------------------------------------------------------------------------------------------------------------
output_dir: "output/train/edgeyolo_tiny_coco_aquarium"        # all train output file will save in this dir
save_checkpoint_for_each_epoch: true                 # save models for each epoch (epoch_xxx.pth, not only best/last.pth)
log_file: "log.txt"                                  # log file (in output_dir)

# dataset & dataloader--------------------------------------------------------------------------------------------------
dataset_cfg: "params/dataset/coco_aquarium.yaml"              # dataset config
batch_size_per_gpu: 8                                # batch size for each GPU
loader_num_workers: 4                                # number data loader workers for each GPU
num_threads: 1                                       # pytorch threads number for each GPU

# the rest--------------------------------------------------------------------------------------------------------------
```

You will also need to set up the dataset configuration file `params/dataset/coco_aquarium.yaml`:

```yaml
type: "coco"

dataset_path: "<...>/Downloads/edgeyolo/Aquarium Combined.v2-raw-1024.coco"

kwargs:
  suffix: "jpg"
  use_cache: true      # (test on i5-12490f) Actual time cost:  52s -> 10s(seg enabled) and 39s -> 4s (seg disabled)

train:
  image_dir: "<...>/Downloads/edgeyolo/Aquarium Combined.v2-raw-1024.coco/train"
  label: "<...>/Downloads/edgeyolo/Aquarium Combined.v2-raw-1024.coco/train/_annotations.coco.json"

val:
  image_dir: "<...>/Downloads/edgeyolo/Aquarium Combined.v2-raw-1024.coco/valid"
  label: "<...>/Downloads/edgeyolo/Aquarium Combined.v2-raw-1024.coco/valid/_annotations.coco.json"

test:
  test_dir: "test2017"

segmentaion_enabled: false

names: ["creatures", "fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"]
```

## Step 4: Train the model

You can train the model using the following command:

```bash
python train.py -c params/train/train_coco_aquarium.yaml
```

This may take some time depending on the dataset size and the model you are using.
Best to have a GPU for training.

## Step 5: Convert the model to ONNX

After training the model, you can convert it to ONNX format using the `export.py` script from EdgeYOLO.

```bash
python export.py --weights output/train/edgeyolo_tiny_coco_aquarium/last.pth --onnx-only 
```

You will find the ONNX model in the `output/export/` folder.

## Step 6: Use the model with OBS Detect

TBD
