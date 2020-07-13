# Image Inpainting Server

The server code for our generative inpainting model. The client code can be found on our [inpainting-client repository](https://github.com/ndarr/image-inpainting-client). It can also be tested online on [https://ndarr.github.io/image-inpainting-client/](https://ndarr.github.io/image-inpainting-client/). 

## Model Architecture
We build the model based on the U-Net architecture by [Ronneberger et al.](https://arxiv.org/pdf/1505.04597.pdf%29和%5bTiramisu%5d%28https://arxiv.org/abs/1611.09326.pdf) with the alterations suggested by [Lui et al.](https://openaccess.thecvf.com/content_ECCV_2018/papers/Guilin_Liu_Image_Inpainting_for_ECCV_2018_paper.pdf). The model was trained on public domain cat pictures. The model parameter file can be found on [Google Drive](https://drive.google.com/file/d/11h1kK2SJ7msQqBQOHUKAMZBmhIq_KSpG/view?usp=sharing).

## Authors

* **Nicolas Darr** - [ndarr](https://github.com/ndarr)
* **Phillip Rust** - [xplip](https://github.com/xplip)
* **Alexander Müller** - [AlexanderMlr](https://github.com/AlexanderMlr)

## Acknowledgments

Thanks to NVIDIA for providing the partialconv2d module for PyTorch: [https://github.com/NVIDIA/partialconv](https://github.com/NVIDIA/partialconv)