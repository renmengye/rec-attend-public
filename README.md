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

## CREMI Experiments
**Note**: Unlock h5_lock:  
`export HDF5_USE_FILE_LOCKING='FALSE'`

### Crop data:  
Run `cremi_Prepare_Eval.ipynb`
### Setup data:
**Configure** the size `opt` in `setup_cvppp.py`.  
**Note**: `setup_cvppp.py` will automatically resize to the size `opt` in `setup_cvppp.py`.    
**Run** `setup_cvppp.py`  


###Run experiments:
+ **Configure** the setting:  
    + Class `TrainArgsParser` in `cmd_args_parser.py` 
    + Number of object `kCvpppNumObj` in `cmd_args_parser.py` 
    + `steps_per_valid` in `cmd_args_parser.py` 
    + `steps_per_trainval` in `cmd_args_parser.py` 
    + `steps_per_plot` in `cmd_args_parser.py` 
    + `num_batch_valid` in `cmd_args_parser.py`
    + `MAX_NUM_ITERATION` in `hungarian.cc` 
    
+ **Choose GPU_id** in: 
    + `box_model_train.py` 
    + `box_model_read.py`
    + `full_model_train.py`  
    + `full_model_eval.py`  
    
+ **Comment** those code in `box_model_train.py`
    ```
      # if 'attn' in self.loggers:
      #   pu.plot_double_attention(
      #       self.loggers['attn'].get_fname(),
      #       x,
      #       results['ctrl_rnn_glimpse_map'],
      #       max_items_per_row=max_items)
    ```
**Run**
```bash
./run_cremi.sh
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
