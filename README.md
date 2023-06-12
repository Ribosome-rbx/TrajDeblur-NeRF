# Deblur NeRF with Trajectory for project of ETHZ 3DVison Lecture
[Report](link) | [Video](https://youtube.com/playlist?list=PLUffCQyBEYtbOQg4-66ZrcuNmsX0OXVKv)
![](https://github.com/Ribosome-rbx/TrajDeblur-NeRF/blob/main/image/deblur_pipeline.png)

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

## Quick Start:
### 1. Setup Environment
* Follow the official [tutorial](https://github.com/limacv/Deblur-NeRF#quick-start) for environment building.
* The list of dependencies could be found at [./requirements.txt](https://github.com/Ribosome-rbx/TrajDeblur-NeRF/blob/main/requirements.txt)
#### if running on euler 
* Load module: `module load gcc/8.2.0 cuda/11.6.2 python/3.8.5 cudnn/8.0.5 cmake/3.19.8 eth_proxy`
* Activate virtual environment: `source ../env-3dvision/bin/activate`

### 2. training and testing
overall procedures please follow [instructions](https://github.com/limacv/Deblur-NeRF#3-setting-parameters) of Deblur-Nerf. We added the following new parameters to construct `config.txt`:

1. kernel_quater_embed --the dim of quaternion coordinate embedding, generally set into 0 or 2.
2. kernel_velocity_embed --the dim of velocity coordinate embedding, generally set into 0 or 2.

### 3. Useful Commands for Running on Euler
* Debug `srun --time=1:30:00 --gpus=1 --gres=gpumem:16g -n 2 --mem-per-cpu=8g --pty bash`
* Check the status of allocation: `squeue` or `watch -n 0 squeue`

* Submit Job: `sbatch --time=4:00:00 --gpus=1 --gres=gpumem:16g -n 2 --mem-per-cpu=8g --output=./logs/raw_output --open-mode=append --wrap="[...cmd...]"`
* Check details of the job: `myjobs -j job_id`
* Check details of the job: `scancel job_id`

* Change access permission for others: `chmod -R u+rwx,g+rwx,o+rx ./`

Training
```sbatch --time=16:00:00 --gpus=1 --gres=gpumem:32g --cpus-per-task=1 --mem-per-cpu=32g --output=./logs/raw_output --open-mode=append --wrap="python run_nerf.py --config configs/demo_blurball.txt > ./logs/training_log"```

Only Render
```sbatch --time=16:00:00 --gpus=1 --gres=gpumem:32g --cpus-per-task=1 --mem-per-cpu=32g --output=./logs/raw_output --open-mode=append --wrap="python run_nerf.py --config configs/demo_blurball.txt --render_only > ./logs/testing_log"```

#### Other useful commands
* upload file on to server: `scp -r /path/filename borong@euler.ethz.ch:/path`
* PNG 2 Video `ffmpeg -framerate 25 -i "%03d.png" -c:v libx264 -pix_fmt yuv420p video.mp4`


## Dataset
### 1. Deblur-NeRF Motino Blur dataset
Follow the instructions [here](https://github.com/limacv/Deblur-NeRF#2-download-dataset) to use original dataset of Deblur-NeRF.
### 2. HoloLens Room Dataset
This dataset is captured by Hololens2 and consists of two video recordings of two different room scenes. Each capture contains thousands of RGB video frames in 1280×720, monocular depth frames in a lower capturing frequency, the intrinsic parameters of the camera, and the corresponding camera poses and timestamp for each RGB frame. For the first capture ([AnnaTrain](https://drive.google.com/file/d/1ejI0oGDvouf8kSXmtE2YtDnUD5xQ9CJ0/view)/[GowthamTrain](https://drive.google.com/file/d/1SDoMu82SKCXeIN0Jx5hPdFrSIh5NdLd5/view)), the HoloLens has a relatively slow movement, which results in a dataset containing less motion blur. While the second capture (named [AnnaTest](https://drive.google.com/file/d/1GM86hnksWmncO_VzHofgo8cX0_KKEzvO/view)/[GowthamTest](https://drive.google.com/file/d/1ch8T6YyFJjmdYxV6ZIc7_MvTgNo4QHTE/view)) contains more motion blur. Here use the AnnaTrain as an example.
```
AnnaTrain
     ├── Depth
     ├── Head
     ├── SceneUnderstanding
     ├── Video
     └── poses_bounds.npy
```


Following these steps to use Room Dataset
1. Run `./llff_convertion.py` to transform camera poses from HoloLens to COLMAP. Store the poses_bounds.npy following the above data structure.
2. Change code in `load_llff.py` line **268** (you'll see instructions there)
3. Modify data paths in config files correspondingly. 

## Contact
Boxiang Rong - borong@ethz.ch

## Citation and Acknowledgmentgs:
Our pipeline is build upon the [Deblur-Nerf](https://github.com/limacv/Deblur-NeRF) pipeline. If you use our code, please make sure to cite the original Deblur-Nerf paper:
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
