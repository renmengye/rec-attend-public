# Set up cityscapes experiments.
# Author: Mengye Ren (mren@cs.toronto.edu)

#########################################################
# Set up folder paths.
# Change lines below.
# Path to KITTI dataset.
KITTI_DATA=/ais/gobi4//mren/data/kitti/object
# Path to model storage.
SAVE_FOLDER=/ais/gobi5/mren/results/rec-attend
# Path to log storage.
DASHBOARD_LOGS=/u/mren/public_html/results
#########################################################

mkdir -p data

if [ ! \( -e "${data/kitti}" \) ]
then
  ln -s $KITTI_DATA data/kitti
fi

FF='results'
if [ ! \( -e "${FF}" \) ]
then
  SAVE_FOLDER=/ais/gobi5/mren/results/rec-attend
  mkdir -p $SAVE_FOLDER
  ln -s $SAVE_FOLDER results
fi

FF='logs'
if [ ! \( -e "${FF}" \) ]
then
  DASHBOARD_LOGS=/u/mren/public_html/results
  ln -s $DASHBOARD_LOGS logs
fi
./setup_kitti.py
