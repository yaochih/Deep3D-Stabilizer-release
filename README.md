# Deep3D Stabilizer (CVPR 2021)
This repository contains the pytorch implementations of the Deep3D Stabilizer. 
> 3D Video Stabilization with Depth Estimation by CNN-based Optimization \
> Yao-Chih Lee, Kuan-Wei Tseng, Yu-Ta Chen, Chien-Cheng Chen, Chu-Song Chen, Yi-Ping Hung \
> **CVPR** 2021 [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Lee_3D_Video_Stabilization_With_Depth_Estimation_by_CNN-Based_Optimization_CVPR_2021_paper.html "Paper")] [[Project Page](https://yaochih.github.io/deep3d-stabilizer.io/ "Project Page")] [[Video](https://www.youtube.com/watch?v=pMluFVA7NDQ)]

[![teaser](https://yaochih.github.io/deep3d-stabilizer.io/img/pipeline.png)](https://www.youtube.com/watch?v=pMluFVA7NDQ)

### Contribution
- The first 3D-based CNN method for video stabilization without training data.
- Handle parallax effect more properly leveraging 3D motion model.
- Allow users to manipulate the stability of a video efficiently.

---
### Setup
- **Main program**
	- Python3.5+ and Pytorch 1.4.0+
	- Other dependencies
 `apt-get install ffmpeg`
 `pip3 install opencv-python scipy tqdm path imageio scikit-image pypng`

- **PWC-Net** for getting optical flow as preprocessing
	- Require Pytorch 0.2.0 with python2.7 & cuda 8 
(please refer to the official [PWC-Net](https://github.com/NVlabs/PWC-Net/tree/master/PyTorch) and its [issue](https://github.com/NVlabs/PWC-Net/issues/76#issuecomment-513803385))

### Running
To test your own video to be stabilized, run the commands below. The stabilized video will be saved in ```outputs/test``` by default.
```bash
python3 geometry_optimizer.py [your-video-path] [--name default=test]
python3 rectify.py [your-video-path] [--name same-as-above] [--stability default=12]
```

#### Stability Manipulation
If you have run ```geometry_optimizer.py``` for the video, you may run ```rectify.py``` for the **same** video multiple times with different ```--stability``` to manipulate the stability efficiently.

---
### Citation
```Bibtex
@InProceedings{Lee_2021_CVPR
    author    = {Lee, Yao-Chih and Tseng, Kuan-Wei and Chen, Yu-Ta and Chen, Chien-Cheng and Chen, Chu-Song and Hung, Yi-Ping},
    title     = {3D Video Stabilization with Depth Estimation by CNN-based Optimization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {10621-10630}
}
```
### License
The provided implementation is strictly for academic purposes only.

### Acknowledgement
We thank the authors for releasing [SC-SfMLearner](https://github.com/JiawangBian/SC-SfMLearner-Release), [monodepth2](https://github.com/nianticlabs/monodepth2), and [PWC-Net](https://github.com/NVlabs/PWC-Net/tree/master/PyTorch).
