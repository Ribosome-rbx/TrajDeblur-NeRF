# Deblur NeRF with Trajectory for project of ETHZ 3DVison Lecture
[Report](link) | [Video](https://youtube.com/playlist?list=PLUffCQyBEYtbOQg4-66ZrcuNmsX0OXVKv)
![](link:pipeline_image)

## Experimental Results: 
./logs/exp/ --experiment outputs
- bookshelf/20230515_2220 --bookshelf scene
- poster/20230515_1450 --poster scene 
- 520v_room/20230516 --room scene 
- poster_less/20230608_1545 --poster scene with less training images 

./data/experiments --experiment data
- bookshelf --bookshelf scene
- poster --poster scene 
- 520view_room --room scene 
- poster_less --poster scene with less training images 

## pipeline:
### 1. load dependencies
* If you are not running on Euler, please consult the official [tutorial](https://github.com/dunbar12138/DSNeRF#dependencies) for environment building.
* The list of dependencies could be found at [./requirements.txt](https://github.com/Dzl666/3DVision_DSNerf/blob/master/requirements.txt)

#### if running on euler 
* Load module: `module load gcc/8.2.0 cuda/11.6.2 python/3.8.5 cudnn/8.0.5 cmake/3.19.8 eth_proxy`
* Activate virtual environment: `source ../env-3dvision/bin/activate`

### 2. run colmap
Note: This part should be done in your local machine 
* download COLMAP and run imgs2poses.py according to the [tutorial](https://github.com/dunbar12138/DSNeRF#generate-camera-poses-and-sparse-depth-information-using-colmap-optional) in the DS-Nerf repo
* in colmap_wrapper.py, you could tailer SiftExtraction.peak_threshold and SiftExtraction.edge_threshold according to your dataset

### 3. training and testing
* overall procedures please consolt the [tutorial](https://github.com/dunbar12138/DSNeRF#how-to-run) for DS-Nerf.
* when constructing config.txt, we have added the following new parameters:
1. seg_video --how many segments to divide the rendered video (tailor according to your GPU capacity)
2. render_itp_nodes --When render_train=True, we allow interpolations between each rendering poses to form smooth but longer videos. render_itp_node refers to the number of Slerp interpolation nodes between each rentering pose when generating the video poses.

#### If running on euler
The following command examples would be useful:
* Debug `srun --time=1:30:00 --gpus=1 --gres=gpumem:16g -n 2 --mem-per-cpu=8g --pty bash`
* Check the status of allocation: `squeue`

* Submit Job: `sbatch --time=4:00:00 --gpus=1 --gres=gpumem:16g -n 2 --mem-per-cpu=8g --output=./logs/raw_output --open-mode=append --wrap="[...cmd...]"`
* Check details of the job: `myjobs -j job_id`
* Check details of the job: `scancel job_id`

* Change access permission for others: `chmod -R u+rwx,g+rwx,o+rx ./`

> Training `sbatch --time=6:00:00 --gpus=1 --gres=gpumem:25g -n 3 --mem-per-cpu=8g --output=./logs/raw_output_poster_less_cont --open-mode=append --wrap="python run_nerf.py --config configs/exp_poster_less.txt > ./logs/training_log_poster_less_cont"`

> Only Render `sbatch --time=4:00:00 --gpus=1 --gres=gpumem:35g -n 5 --mem-per-cpu=8g --output=./logs/raw_output_bookshelf --open-mode=append --wrap="python run_nerf.py --config configs/exp_bookshelf.txt --render_only > ./logs/rendering_log_bookshelf"`


## 4. other pipelines 
Here are documents and comments related to our pipeline for converting the point cloud generated from rgbd alingment to the depth information for training DS-Nerf and using hololens coordinates instead of COLMAP estimations. **Important: these pipelines are currently unable to produce desired results due to the mismatch between the coordinates of our generated depth map/camera poses and the coordiantes expected for DS-Nerf**
* ./open3d_llff/color_map_optimization.py --aligns a set of depth map frames together with a set of RGB image frames. Please modify the input parameters at line 169, where instructiions could also be found. 
Output: a set of aligned depth maps and a list containing the paths to the RGB frames corresponding to each generated depth frames. 
* ./open3d_llff/llff_conversion.py --converts the Hololens poses, intrinsics, and the color_map_optimization.py outputs into the format suitable for the "colmap_llff" input of DS-Nerf training (see line 437 in ren_nerf.py). More instructions coule be found inside the code. 
* ./open3d_llff/poses_from_colmap.py --converts the poses_bounds.npy output of colmap into the training/testing posrs abd bounds of 'colmap_llff' form.

## Contact
Ziyao Shang - zshang@ethz.ch
Zilong Deng - dengzi@ethz.ch

## Citation and Acknowledgmentgs:
Our pipeline is build upon the [DS-Nerf](https://github.com/dunbar12138/DSNeRF) pipeline. If you use our code, please make sure to cite the original DS-Nerf paper:
```
@InProceedings{kangle2021dsnerf,
    author    = {Deng, Kangle and Liu, Andrew and Zhu, Jun-Yan and Ramanan, Deva},
    title     = {Depth-supervised {NeRF}: Fewer Views and Faster Training for Free},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022}
}
```


## Euler commands
* Load module`module load gcc/8.2.0 cuda/11.6.2 python/3.8.5 cudnn/8.0.5 cmake/3.19.8 eth_proxy`

* upload file on to server: `scp -r /path/filename borong@euler.ethz.ch:/path`

* If using virtual environment: `source ../env-3dvision/bin/activate`

* Debug with Euler `srun --time=1:00:00 --gpus=1 --gres=gpumem:30g -n 4 --mem-per-cpu=8g --pty bash`

* Check details of the job `myjobs -j job_id`
* Check queue: `watch -n 0 squeue`
* Change access `chmod -R u+rwx,g+rwx,o+rx ./`

> Training `sbatch --time=16:00:00 --gpus=1 --gres=gpumem:32g --cpus-per-task=1 --mem-per-cpu=32g --output=./logs/raw_output --open-mode=append --wrap="python run_nerf.py --config configs/demo_blurball.txt > ./logs/training_log"`

> PNG 2 Video `ffmpeg -framerate 25 -i "%03d.png" -c:v libx264 -pix_fmt yuv420p rgb.mp4`





This is the official implementation of the paper [Deblur-NeRF: Neural Radiance Fields from Blurry Images](https://arxiv.org/abs/2111.14292)
# Deblur-NeRF

Deblur-NeRF is a method for restoring a sharp NeRF given blurry multi-view input images. It works for camera motion blur, out-of-focus blur, or even object motion blur. If you are interested, please find more information on the website [here](https://limacv.github.io/deblurnerf/).

![](https://limacv.github.io/deblurnerf/images/teaser.jpg)

## Method Overview

![](https://limacv.github.io/deblurnerf/images/pipeline.png)
When rendering a ray, we first predict N sparse optimized rays based on a canonical kernel along with their weights. After rendering these rays, we combine the results to get the blurry pixel. During testing, we can directly render the rays without kernel deformation resulting in a sharp image.

## Quick Start

### 1. Install environment

```
git clone https://github.com/limacv/Deblur-NeRF.git
cd Deblur-NeRF
pip install -r requirements.txt
```
<details>
  <summary> Dependencies (click to expand) </summary>

   - numpy
   - scikit-image
   - torch>=1.8
   - torchvision>=0.9.1
   - imageio
   - imageio-ffmpeg
   - matplotlib
   - configargparse
   - tensorboardX>=2.0
   - opencv-python
</details>

### 2. Download dataset
There are total of 31 scenes used in the paper. We mainly focus on camera motion blur and defocus blur, so we use 5 synthetic scenes and 10 real world scenes for each blur type. We also include one case of object motion blur. You can download all the data in [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/lmaag_connect_ust_hk/EqB3QrnNG5FMpGzENQq_hBMBSaCQiZXP7yGCVlBHIGuSVA?e=UaSQCC). 

For a quick demo, please download ```blurball``` data inside ```real_camera_motion_blur``` folder.

### 3. Setting parameters
Changing the data path and log path in the ```configs/demo_blurball.txt```

### 4. Execute

```
python3 run_nerf.py --config configs/demo_blurball.txt
```
This will generate MLP weights and intermediate results in ```<basedir>/<expname>```, and generate tensorboard files in ```<tbdir>```

## Some Notes

### GPU memory
One drawback of our method is the large memory usage. Since we need to render multiple rays to generate only one color, we require a lot more memory to do that. We train our model on a ```V100``` GPU with 32GB GPU memory. If you have less memory, setting ```N_rand``` to a smaller value, or use multiple GPUs.

### Multiple GPUs
you can simply set ```num_gpu = <num_gpu>``` to use multiple gpus. It is implemented using ```torch.nn.DataParallel```. We've optimized the code to reduce the data transfering in each iteration, but it may still suffer from low GPU usable if you use too much GPU.

### Optimizing the CRF
As described in the paper, we use gamma function to model the nonlinearity of the blur kernel convolution (i.e. CRF). We provide an optional solution to optimize a learnable CRF function that transfer the color from linear space to the image color space. We've found that optimizing a learnable CRF sometimes achieves better visual performance. 

To make the CRF learnable, change the config to ```tone_mapping_type = learn```.

### Rollback to the original NeRF
By setting ```kernel_type = none```, our implementation runs the original implementation of NeRF.


### Generate synthetic dataset
We released the raw blender files to generate synthetic dataset in the dataset link above.
Running the script in the blender files gives you synthetic data format.
To convert them to the llff data format, run python script in ```scripts/synthe2poses.py```. (Some path variables need to be changed)

### Your own data
The code digests the same data format as in the original implementation of NeRF (with the only difference being the images maybe blurry). So you can just use their data processing scripts to generate camera poses to ```poses_bounds.npy```.

Since all of our datasets are of type "llff" (face forward dataset), we do not support other dataset_type. But it should be relatively easy to migrate to other dataset_type. 

## Limitation
Our method does not work for consistent blur. For example, in the defocus blur case, if all the input views focus on the same foreground, leaving the background blur, our method only gives you NeRF with sharp foreground and blur background. Please check our paper for the definition of consistent blur. This is left for future work.

## Citation
If you find this useful, please consider citing our paper:
```
@misc{li2022deblurnerf,
    title={Deblur-NeRF: Neural Radiance Fields from Blurry Images},
    author={Ma, Li and Li, Xiaoyu and Liao, Jing and Zhang, Qi and Wang, Xuan and Wang, Jue and Pedro V. Sander},
    year={2021},
    eprint={2111.14292},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Acknowledge
This source code is derived from the famous pytorch reimplementation of NeRF, [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch/). We appreciate the effort of the contributor to that repository.
