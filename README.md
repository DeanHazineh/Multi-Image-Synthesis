# Polarization Multi-Image Synthesis with Birefringent Metasurfaces

###  [ [Project Website] ](https://deanhazineh.github.io/publications/Multi_Image_Synthesis/MIS_Home.html) [ [Paper and Supplement] ](Documents/combined_paper.pdf)
<p>This paper was published in the proceedings of the 2023 IEEE International Conference on Computational Photography (ICCP).</p>

<p>Along with this repository which contains the code to reproduce the results in the main paper, we also release a seperate package, called <a href="https://github.com/DeanHazineh/DFlat" target="_blank">D-Flat</a>, which provides a comprehensive auto-differentiable framework for the design of diffractive optics.</p>

<p>If you would like to reference this work or if you make use of the code/dataset, please cite it using the format shown at the bottom: </p>

## Project Summary:
<div align="center">
  <img src="Documents/featured.png" alt="Featured Image" width="90%">
</div>

<p>The heart of this project is an investigation into the topic of multi-coded imaging, where a (fixed) camera captures multiple, coded images of a scene in a single snapshot. These images can then be passed to a computational back-end for task-specific digital processing.</p> 

<p>Proposed camera designs in the past have often captured a set of images at once by using a wavelength-dependent optical mask at the lens plane in conjunction with an array of spectral filters tiled above the photosensor pixels (e.g. a Bayer color filter). In this work, we propose an alternate design paradigm and demonstrate a new architecture (shown in the left image) where distinct coded images are captured on different polarization states.</p>

<p>We utilize a metasurface composed of carefully engineered nanostructures as both the spatial modulation pattern and the focusing lens. This optic produces several distinctly coded images of the scene on different polarization channels. These images are then seperately sampled and digitally measured using an off-the-shelf polarization-sensitive photosensor (i.e. a photosensor with different polarization filters above the pixel). This camera enables the simultaneous capture of four coded images at once and presents several key advantages as compared to prior methods which utilize the wavelength of light as information channels. </p>

<p>In this work, we explain how this proposed optical systems can be designed and optimized in an end-to-end fashion, enabling it's usage for a wide range of tasks in computer vision and computational imaging. Notably, in doing so, we also tackle a fundamental problem: In theory, the set of different polarization channels can interferre with each other and this means that the codes imparted on the four images cannot be independently specified. In practice, however, we show that this limitation can be relaxed for a cost and that all four polarization channels can be practically used. </p>

<p> As an example, we demonstrate the use of this architecture for the task of opto-electronic image processing. As shown on the right, four images of a scene are captured by the sensor but each of the four is subtly coded. When the four images are simply added together in post-processing, one obtains a differented rendering of the scene for minimum computation costs. This is the first snapshot, compact (single-lens) implementation of opto-electronic image processing demonstrated to date which can be used on any scene. </p>

## Install and Run
To run the code in this repository, one must first install the python package, D-Flat. All core physics functions like field propagation are stored there. Instructions are re-written here for convenience but see the official repository for more details:

Note that git LFS should be installed if not already via `git lfs install` at the terminal. Then install the python package to your environment via:

```
git clone https://github.com/DeanHazineh/DFlat
python setup.py develop
pip install -r requirements.txt
```

After, you may then download and use this repostiory like normal:
```
git clone https://github.com/DeanHazineh/Multi-Image-Synthesis
```


## Credits and Acknowledgements:
```
@INPROCEEDINGS{Hazineh2023,
  Author = {Dean Hazineh and Soon Wei Daniel Lim and Qi Guo and Federico Capasso and Todd Zickler},
  booktitle = {2023 IEEE International Conference on Computational Photography (ICCP)}, 
  Title = {Polarization Multi-Image Synthesis with Birefringent Metasurfaces},
  Year = {2023},
}
```
