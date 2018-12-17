# Clear Re-ID: Tackling Occlusion in Person re-identification

This is a project that tries to tackle the problem of occluded images for person re-identification. Our model implements the baseline model from Align Re-ID and modifies the loss and adds a fully connected layer to perform better on occluded images.

## Installation

We use Python 2.7 and Pytorch 0.3. For installing Pytorch, follow the [official guide](http://pytorch.org/). Other packages are specified in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Dataset

We use Market1501 dataset. Download the dataset from [Google Drive](https://drive.google.com/open?id=1CaWH7_csm9aDyTVgjs7_3dlZIWqoBlv4). This is a formated dataset derived from the original Market1501 dataset. This can also be created by running the following command on the original Market1501 dataset that can be downloaded from [here](http://www.liangzheng.org/Project/project_reid.html).

```bash
python 	AlignedReID-Re-Production-Pytorch/script/dataset/transform_market1501.py \
--zip_file ~/Dataset/market1501/Market-1501-v15.09.15.zip \
--save_dir ~/Dataset/market1501
```

## Occlusion Simulator
To generate occluded images for the Market1501 dataset, use the following command. It puts the images in /transformed folder inside the source image folder.

```bash
python utils/occlusion_simulator.py 
  --d '~/Dataset/market1501/images/' 
  --h 50 
  --w 50 
  --m 'random'
  --partition '/home/ubuntu/Dataset/market1501/partitions.pkl'
```
## Examples

### `ResNet-50 + Global Loss + OBC Loss` on Market1501

Our model works, as of now, only on Market dataset. Use the following command 

```bash
python AlignedReID-Re-Production-Pytorch/script/experiment/train.py \
-d '(0,)' \
-r 1 \
--dataset market1501 \
--ids_per_batch 16 \
--ims_per_id 4 \
--normalize_feature false \
-gm 0.3 \
-glw 0.8 \
-llw 0 \
-idlw 0 \
-obclw 0.2 \
--base_lr 2e-4 \
--lr_decay_type exp \
--exp_decay_at_epoch 151 \
--total_epochs 300
```

### Log

During training, you can run the [TensorBoard](https://github.com/lanpa/tensorboard-pytorch) and access port `6006` to watch the loss curves etc. E.g.

```bash
# Modify the path for `--logdir` accordingly.
tensorboard --logdir YOUR_EXPERIMENT_DIRECTORY/tensorboard
```
For help regarding TensorBoard.

```bash
tensorboard --help
```

### Training Time

Using AWS p2.xlarge instances with 1 GPU took over 15 hours to complete the training.

### Testing Time
To test on the Market1501 images (both occluded and non-occluded) it takes ~10 mins on the same AWS instance.
For more usage of TensorBoard, see the website and the help:

# References & Credits

[1] G. Wang, Y. Yuan, X. Chen, J. Li, and X. Zhou, “Learning discriminative features with multiplegranularities for person re-identification,” 2018.

[2] J. Zhuo, Z. Chen, J. Lai, and G. Wang, “Occluded person re-identification,”arXiv preprintarXiv:1804.02792, 2018.

[3]D. Li, X. Chen, Z. Zhang, and K. Huang, “Learning deep context-aware features over body andlatent parts for person re-identification,” inProceedings of the IEEE Conference on ComputerVision and Pattern Recognition, pp. 384–393, 2017.6
 
 [4]H. Zhao, M. Tian, S. Sun, J. Shao, J. Yan, S. Yi, X. Wang, and X. Tang, “Spindle net: Personre-identification with human body region guided feature decomposition and fusion,”2017 IEEEConference on Computer Vision and Pattern Recognition (CVPR), pp. 907–915, 2017.
 
 [5]S. Bak and P. Carr, “Person re-identification using deformable patch metric learning,”2016 IEEEWinter Conference on Applications of Computer Vision (WACV), pp. 1–9, 2016.
 
 [6]W. Chen, X. Chen, J. Zhang, and K. Huang, “Beyond triplet loss: a deep quadruplet network forperson re-identification,” inThe IEEE Conference on Computer Vision and Pattern Recognition(CVPR), vol. 2, 2017.
 
 [7]X.  Zhang,  H.  Luo,  X.  Fan,  W.  Xiang,  Y.  Sun,  Q.  Xiao,  W.  Jiang,  C.  Zhang,  and  J.  Sun,“Alignedreid: Surpassing human-level performance in person re-identification,”arXiv preprintarXiv:1711.08184, 2017.
 
 [8]A. Hermans, L. Beyer, and B. Leibe, “In defense of the triplet loss for person re-identification,”2017.
 
 [9]L. Zheng, L. Shen, L. Tian, S. Wang, J. Wang, and Q. Tian, “Scalable person re-identification: Abenchmark,”2015 IEEE International Conference on Computer Vision (ICCV), pp. 1116–1124,2015.

