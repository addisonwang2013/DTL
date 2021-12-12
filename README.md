## Dual Transfer Learning for Event-based End-task Prediction via Pluggable Event to Image Translation (ICCV'21)

We have updated our paper by correcting some typos and adding more references. 

Please refer to the Arxiv paper at https://arxiv.org/pdf/2109.01801.pdf for the latest information.

## Citation
If you find this resource helpful, please cite the paper as follows:

```bash
@inproceedings{wang2021dual,
  title={Dual transfer learning for event-based end-task prediction via pluggable event to image translation},
  author={Wang, Lin and Chae, Yujeong and Yoon, Kuk-Jin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={2135--2145},
  year={2021}
}
```
* Segmentic segmentation on the general visual condition.

![image](https://user-images.githubusercontent.com/79432299/143512905-d07842b0-0348-4378-aead-a2689868e5a6.png)

* Segmentic segmentation on over-exposed visual condition.

![image](https://user-images.githubusercontent.com/79432299/143513108-49acae9c-f217-49a3-91d2-d69a8976341a.png)

* Depth estimation on the night-scene condition

![image](https://user-images.githubusercontent.com/79432299/143513185-4bc9b81c-725f-4f95-a9e5-a29646bf4c7b.png)


## Setup

Download 

``` python
git clone  https://github.com/addisonwang2013/DTL/
```

Make your own environment

```python
conda create -n myenv python=3.7
conda activate myenv
```

Install the requirements

```bash
cd evdistill

pip install -r requirements.txt
```

Download example validation data (general and LDR visual condtions) from this link: [DDD17 example data](https://drive.google.com/drive/u/2/folders/1Q-1djBTjc8vbaDBLtfmSXZ1W5lzirz5p)

* For DDD17 dataset general visiual condition, please put the dataset to `./dataset/ddd17/general/`
* For DDD17 dataset low dynamic range (LDR) condition, please put the dataset to `./dataset/ddd17/ldr/`

Download the pretrained models from this link: [checkpoints](https://drive.google.com/drive/u/2/folders/1j6Xu5iO9QJLG_BYHHYdpErpcHM9rwFWD)

* Put the checkpoint of event segmentation network into `./res/`

Modify the ``` python configurations.py ``` in the `configs` folder with the relevant paths to the test data and checkpoints

* For the test data, *e.g.* DDD17, please assign the path to `./test_dir/ddd17/`
* For the checkpoint of event network, please assign the path to `./res/eventdual_best.pth`


Visualizing semantic segmentation results for general and LDR visual condtions:

```python
python visualize.py
```

## Note 

In this work, for convenience, the event data are embedded and stored as multi-channel event images, which are the paired with the aps frames. It is also possible to directly feed event raw data after embedding to the student network directly with aps frames.

## Acknowledgement
The skeleton code is inspired by [Deeplab-v3-Plus](https://github.com/CoinCheung/DeepLab-v3-plus-cityscapes/) and EDSR (https://github.com/sanghyun-son/EDSR-PyTorch)

## License
[MIT](https://choosealicense.com/licenses/mit/)


