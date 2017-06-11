# Set up cityscapes experiments.
# Author: Mengye Ren (mren@cs.toronto.edu)

#########################################################
# Set up folder paths.
# Change lines below.
# Path to cityscapes dataset.
CTY_DATA=/ais/gobi4//mren/data/cityscapes
# Path to model storage.
SAVE_FOLDER=/ais/gobi5/mren/results/rec-attend
# Path to log storage.
DASHBOARD_LOGS=/u/mren/public_html/results
# Path to pretrained model outputs.
PRETRAINED_LRR_PATH=/ais/gobi4/mren/models
#########################################################

mkdir -p data

if [ ! \( -e "${data/cityscapes}" \) ]
then
  ln -s $CTY_DATA data/cityscapes
fi

FF='results'
if [ ! \( -e "${FF}" \) ]
then
  mkdir -p $SAVE_FOLDER
  ln -s $SAVE_FOLDER results
fi

FF='logs'
if [ ! \( -e "${FF}" \) ]
then
  ln -s $DASHBOARD_LOGS logs
fi

./setup_cityscapes.py

# Download semantic segmentation outputs from G. Ghiasi and C. C. Fowlkes.
# Laplacian pyramid reconstruction and refinement for semantic segmentation. 
# In ECCV, 2016:
wget http://www.cs.toronto.edu/~mren/recattend/LRR.zip $PRETRAINED_LRR_PATH/LRR.zip
unzip $PRETRAINED_LRR_PATH/LRR.zip $PRETRAINED_LRR_PATH
mkdir -p pretrained
ln -s $PRETRAINED_LRR_PATH/LRR pretrained/LRR
