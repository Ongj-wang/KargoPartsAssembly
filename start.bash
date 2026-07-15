# Source conda setup (no need to run conda init)
source /home/user/miniforge3/etc/profile.d/conda.sh
conda activate yolo_jaka
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH

python3 Main.py
