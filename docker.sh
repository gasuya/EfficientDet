docker run --gpus all -it --rm \
-v $(pwd):/work \
-p 8887:8887 \
--ipc=host \
-w /work \
--name adachi \
efficientdet_pytorch

pip install matplotlib
pip install opencv-python
pip install webcolors
pip install xmltodict
pip install pyyaml
pip install Cython
pip install pycocotools
pip install tqdm
pip install tensorboardX
pip uninstall numpy
pip install numpy==1.17

python train.py -c 0 -p hogehoge --batch_size 1 --lr 1e-5 --num_epochs 100 \
 --load_weights weights/efficientdet-d0.pth \
 --head_only True

 python train.py -c 4 -p hogehoge --batch_size 1 --lr 1e-5 --num_epochs 10 \
  --load_weights weights/efficientdet-d4.pth \
  --head_only True

python train.py -c 0 -p hogehoge --batch_size 1 --lr 1e-5 --num_epochs 10 --debug True

python coco_eval.py -c 0 -p hogehoge -w logs/hogehoge/efficientdet-d0_4_90.pth

cd Yet-Another-EfficientDet-Pytorch
python train.py -c 0 -p shape --head_only True --lr 1e-3 --batch_size 1 --load_weights weights/efficientdet-d1.pth  --num_epochs 50

python efficientdet_test.py -c 0 -p shape -w logs/shape/efficientdet-d0_49_1400.pth


nvidia-docker run --rm -it\
        --user root\
        -e NVIDIA_VISIBLE_DEVICES=0\
        -e NB_UID=$UID \
        --ipc=host \
        -p 58888:8888 -p 50027:22 -p 6000:6000 \
        -v $(pwd):/work \
        -w /work \
        --name notebook \
        jupyter/datascience-notebook bash
        #registry:5000/jupyter/toyo-notebook:tf-1.12.0-cuda9.0-cudnn7-deval-ubuntu1604 bash
