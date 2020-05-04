# DML_HistoImgRetrieval
We proposed a deep metric learning approach for histopathological image retrieval.

This work has been presented in BIBM2019 conference in an oral talk during 18-21 November, San Diego, CA, USA.

![framework](https://github.com/easonyang1996/DML_HistoImgRetrieval/blob/master/figs/framework.jpeg)

We construct a deep neural network based on the mixed attention mechanism to learn an embedding function under the supervision of image category information. We evaluate the proposed method on two datasets, one is our self-established dataset and the other is Kimia Path24<sup>[1]</sup>. The visualization of the retireval results on our self-established dataset is shown below. (a) shows successful retrieval results and (b) shows failed retrieval results.

![Result](https://github.com/easonyang1996/DML_HistoImgRetrieval/blob/master/figs/result.jpeg)

Due to the privacy policy, We only release the source code of experiments on the public dataset Kimia Path24.

# Requirements
python3

pytorch

torchvision

torchsummary 


# Installation
DML_HistoImgRetrieval can be downloaded by
```
git clone https://github.com/easonyang1996/DML_HistoImgRetrieval.git
```
Installation has been tested on a Linux/MacOS platform.

# Instructions
We provide detailed step-by-step instructions for reproducing experiments of the proposed method on Kimia Path24. You can also run the proposed method on your own dataset in a similar way.

**Step 1** Prepare the dataset.

Please download the dataset at [Kimia Path24](https://kimialab.uwaterloo.ca/kimia/index.php/pathology-images-kimia-path24/) and extract the `.zip` file in the current directory. Then, run `./KIMIA_data/data_process.py` to do preprocessing and move images to `./KIMIA_data/train/` and `./KIMIA_data/test/`.

![KIMIA](https://github.com/easonyang1996/DML_HistoImgRetrieval/blob/master/figs/KIMIA_instance.jpeg)

**Step 2** Train the model.

Run `./train.py` to train the model. There are three models (`ABE_M`, `resnet50`, and `se_resnet50`) and three loss functions (`ABELoss`, `ContrastiveLoss`, and `MSLoss`) can be chosen. The combination of `se_resnet50` and `MSLoss` is corresponding to the proposed method. `./train_traindata.py` can be used to reproduce the experiment about the size of training data.
```
python3 train.py <se_resnet50> <MSLoss>
```

**Step 3** Test the model.

Run `./multi_test.py` to calculate evaluation metrics on all weight files in the given `weight_folder`.
```
python3 multi_test.py <weight_folder>
```
The proposed method achieves the new state-of-the-art on Kimia Path24 retrieval task.

![cfm](https://github.com/easonyang1996/DML_HistoImgRetrieval/blob/master/figs/confusion_mat.jpeg)

# Citation

**Yang, Pengshuai**, Yupeng Zhai, Lin Li, Hairong Lv, Jigang Wang, Chengzhan Zhu, and Rui Jiang. "Liver Histopathological Image Retrieval Based on Deep Metric Learning." In 2019 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), pp. 914-919. IEEE, 2019.

```
@inproceedings{yang2019liver,
  title={Liver Histopathological Image Retrieval Based on Deep Metric Learning},
  author={Yang, Pengshuai and Zhai, Yupeng and Li, Lin and Lv, Hairong and Wang, Jigang and Zhu, Chengzhan and Jiang, Rui},
  booktitle={2019 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={914--919},
  year={2019},
  organization={IEEE}
}
```

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Reference
[1] Babaie M, Kalra S, Sriram A, Mitcheltree C, Zhu S, Khatami A, Rahnamayan S, Tizhoosh HR. Classification and retrieval of digital pathology scans: A new dataset. InProceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops 2017 (pp. 8-16).
