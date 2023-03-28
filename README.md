## Class-Level Confidence Based 3D Semi-Supervised Learning
This is a Pytorch implementation of Class-Level Confidence Based 3D Semi-Supervised Learning.

Paper link: https://arxiv.org/abs/2210.10138

Detection part: https://github.com/Zhimin-C/Confid-SSL-Det/tree/main

### Requirements
python >= 3.7

pytorch >= 1.6

h5py

scikit-learn

and

```shell script
pip install pointnet2_ops_lib/.
```

### Models

The path of the model is in ./checkpoints/best/models/model.t7

Pretrained model: https://drive.google.com/file/d/1PifGItW1m66AusCE4upVqxCabtjZTpgs/view?usp=sharing

### Example training and testing
```shell script
# train
python main.py --exp_name=train --num_points=1024 --use_sgd=True --batch_size 32 --epochs 250 --lr 0.0001

# test
python main.py --exp_name=test --num_points=1024 --use_sgd=True --eval=True --model_path=checkpoints/best/models/model.t7 --test_batch_size 8

```

### Citation
If it is helpful for your work, please cite this paper:
```latex
@inproceedings{chen2023class,
  title={Class-Level Confidence Based 3D Semi-Supervised Learning},
  author={Chen, Zhimin and Jing, Longlong and Yang, Liang and Li, Yingwei and Li, Bing},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={633--642},
  year={2023}
}
```
