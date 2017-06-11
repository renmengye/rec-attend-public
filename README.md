# rec-attend-public
Code that implements paper "End-to-End Instance Segmentation with Recurrent Attention".

## Dependencies
* Python 2.7
* TensorFlow 0.12 (not compatible with TensorFlow 1.0)
* OpenCV
* NumPy
* SciPy
* PyYaml
* hdf5 and H5Py
* tqdm
* Pillow (required by cityscapes evaluation)

## Installation
Compile Hungarian matching module
```bash
./hungarian_build.sh
```

## CVPPP Experiments
First modify `setup_cvppp.sh` with your dataset folder paths.
```bash
./setup_cvppp.sh
```

Run experiments:
```bash
./run_cvppp.sh
```

## KITTI Experiments
First modify `setup_kitti.sh` with your dataset folder paths.
```bash
./setup_kitti.sh
```

Run experiments:
```bash
./run_cvppp.sh
```

## Cityscapes Experiments
First modify `setup_cityscapes.sh` with your dataset folder paths.
```bash
./setup_cityscapes.sh
```

Run experiments:
```bash
./run_cityscapes.sh
```

## Citation
If you use our code, please consider cite the following:
End-to-End Instance Segmentation with Recurrent Attention. Mengye Ren, Richard 
S. Zemel. CVPR 2017.
```
@inproceedings{ren17recattend,
  author    = {Mengye Ren and Richard S. Zemel},
  title     = {End-to-End Instance Segmentation with Recurrent Attention},
  booktitle = {CVPR},
  year      = {2017}
}
```
