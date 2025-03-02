<div align="center">
<h2>High-Fidelity Relightable Monocular Portrait Animation with Lighting-Controllable Video Diffusion Model</h2>

[Mingtao Guo]()<sup>1</sup><sup>†</sup>&nbsp;
[Guanyu Xing]()<sup>1</sup><sup>‡</sup>&nbsp;
[Yanli Liu]()<sup>1,2</sup><sup>▽</sup>&nbsp;


<sup>1</sup> National Key Laboratory of Fundamental Science on Synthetic Vision,
Sichuan University, Chengdu, China 

<sup>2</sup> College of Computer Science, Sichuan University, Chengdu, China 

<sup>†</sup> First author  <sup>‡</sup> Second author  <sup>▽</sup> Corresponding author  

<h3 style="color:#b22222;"> To Appear at CVPR 2025 </h3>

<h4>
<a href="https://arxiv.org/abs/2502.19894">📄 arXiv Paper</a> &nbsp; 
<a href="">🌐 Project Page</a> &nbsp; 
<a href="">📺 Video</a>
</h4>

</div>

<div align="center">
<img src="assets/intro.png?raw=true" width="100%">
</div>
</div>

## :fire: News
* **[2025.03.02]** Our pre-trained model is out on [HuggingFace](https://huggingface.co/MartinGuo/relightable-portrait-animation)!
* **[2025.02.27]** ⭐ Exciting News! Relightable-Portrait-Animation got accepted by CVPR 2025!

## :bookmark_tabs: Todos
We are going to make all the following contents available:
- [x] Model inference code
- [x] Model checkpoint
- [ ] Training code

## Installation

1. Clone this repo locally:

```bash
git clone https://github.com/MingtaoGuo/relightable-portrait-animation
cd relightable-portrait-animation
```
2. Install the dependencies:

```bash
conda create -n relipa python=3.8 -y
conda activate relipa
```

3. Install packages for inference:

```bash
pip install -r requirements.txt
```
## Download weights
```shell
mkdir pretrained_weights
git-lfs install
git clone https://huggingface.co/MartinGuo/relightable-portrait-animation
mv relightable-portrait-animation pretrained_weights/relipa

git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid
mv stable-video-diffusion-img2vid pretrained_weights
```
The weights will be put in the `./pretrained_weights` directory. Heads up! The whole downloading process could take quite a long time.
Finally, these weights should be orgnized as follows:

```text
./pretrained_weights/
|-- relipa
|   |-- unet.pth
|   |-- ref_embedder.pth
|   |-- light_embedder.pth
|   |-- head_embedder.pth
|-- sd-vae-ft-mse
|   |-- config.json
|   |-- diffusion_pytorch_model.bin
|-- stable-video-diffusion-img2vid
    |-- feature_extractor
    |   |-- preprocessor_config.json
    |-- scheduler
    |   |-- scheduler_config.json
    |-- model_index.json
    |-- unet
    |   |-- config.json
    |   |-- diffusion_pytorch_model.safetensors
    |   |-- diffusion_pytorch_model.fp16.safetensors
    |-- image_encoder
    |   |-- config.json
    |   |-- model.safetensors
    |   |-- model.fp16.safetensors
```

## Citation
If you use this model in your research, please consider citing:

```bibtex
@misc{guo2025highfidelityrelightablemonocularportrait,
      title={High-Fidelity Relightable Monocular Portrait Animation with Lighting-Controllable Video Diffusion Model}, 
      author={Mingtao Guo and Guanyu Xing and Yanli Liu},
      year={2025},
      eprint={2502.19894},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.19894}, 
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
