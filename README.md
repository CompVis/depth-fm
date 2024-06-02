<p align="center">
 <!-- <h2 align="center">📻 DepthFM: Fast Monocular Depth Estimation with Flow Matching</h2> -->
 <h2 align="center"><img src=assets/figures/radio.png width=28> DepthFM: Fast Monocular Depth Estimation with Flow Matching</h2>
 <p align="center"> 
    Ming Gui<sup>*</sup> · Johannes S. Fischer<sup>*</sup> · Ulrich Prestel · Pingchuan Ma
 </p><p align="center"> 
    Dmytro Kotovenko · Olga Grebenkova · Stefan A. Baumann · Vincent Tao Hu · Björn Ommer
 </p>
 <p align="center"> 
    <b>CompVis Group @ LMU Munich</b>
 </p>
  <p align="center"> <sup>*</sup> <i>equal contribution</i> </p>
</p>

 </p>
 
 
[![Website](assets/figures/badge-website.svg)](https://depthfm.github.io)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2403.13788)


![Cover](/assets/figures/dfm-cover.png)


## 📻 Overview

We present **DepthFM**, a state-of-the-art, versatile, and fast monocular depth estimation model. DepthFM is efficient and can synthesize realistic depth maps within *a single inference* step. Beyond conventional depth estimation tasks, DepthFM also demonstrates state-of-the-art capabilities in downstream tasks such as depth inpainting and depth conditional synthesis.

With our work we demonstrate the successful transfer of strong image priors from a foundation image synthesis diffusion model (Stable Diffusion v2-1) to a flow matching model. Instead of starting from noise, we directly map from input image to depth map.


## 🛠️ Setup

This setup was tested with `Ubuntu 22.04.4 LTS`, `CUDA Version: 12.2`, and `Python 3.11.5`.

First, clone the github repo...

```bash
git clone git@github.com:CompVis/depth-fm.git
cd depth-fm
```

Then download the weights via

```bash
wget https://ommer-lab.com/files/depthfm/depthfm-v1.ckpt -P checkpoints/
```

Now you have either the option to setup a virtual environment and install all required packages with `pip`

```bash
pip install -r requirements.txt
```

or if you prefer to use `conda` create the conda environment via

```bash
conda env create -f environment.yml
```

Now you should be able to listen to DepthFM! 📻 🎶


## 🚀 Usage

You can either use the notebook `inference.ipynb` or just run the python script `inference.py` as follows

```bash
python inference.py \
   --num_steps 2 \
   --ensemble_size 4 \
   --img assets/dog.png \
   --ckpt checkpoints/depthfm-v1.ckpt
```

The argument `--num_steps` allows you to set the number of function evaluations. We find that our model already gives very good results with as few as one or two steps. Ensembling also improves performance, so you can set it via the `--ensemble_size` argument. Currently, the inference code only supports a batch size of one for ensembling.

## 📈 Results

Our quantitative analysis shows that despite being substantially more efficient, our DepthFM outperforms the current state-of-the-art generative depth estimator Marigold **zero-shot** on a range of benchmark datasets. Below you can find a quantitative comparison of DepthFM against other affine-invariant depth estimators on several benchmarks.

![Results](/assets/figures/sota-comparison.jpg)



## Trend

[![Star History Chart](https://api.star-history.com/svg?repos=CompVis/depth-fm&type=Date)](https://star-history.com/#CompVis/depth-fm&Date)




## 🎓 Citation

Please cite our paper:

```bibtex
@misc{gui2024depthfm,
      title={DepthFM: Fast Monocular Depth Estimation with Flow Matching}, 
      author={Ming Gui, Johannes S. Fischer, Ulrich Prestel, Pingchuan Ma, Dmytro Kotovenko, Olga Grebenkova, Stefan Andreas Baumann, Vincent Tao Hu, Björn Ommer},
      year={2024},
      eprint={2403.13788},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
