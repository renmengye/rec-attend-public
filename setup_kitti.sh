# Change lines below to the path where you store the dataset.
KITTI_DATA=/ais/gobi4//mren/data/kitti/object
mkdir -p data
ln -s $KITTI_DATA data/kitti
# ./setup_kitti.py

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