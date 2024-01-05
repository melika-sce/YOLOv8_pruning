#PBS -N yolov8m_100_vis_1024_8
#PBS -m abe
#PBS -M melika.sce@gmail.com
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l mem=8G
#PBS -q SP

cd $PBS_O_WORKDIR

source /share/apps/Anaconda/anaconda3.8/bin/activate yo8

python train.py > yolov8m_100_vis_1024_8.txt
