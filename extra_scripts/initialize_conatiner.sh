# update apt
apt update -y 
apt install -y sudo 
# install basic packages and python3
sudo apt install -y curl wget nano git x11-apps 

# install packages
sudo apt install -y libgl1-mesa-glx libxrandr2 libxinerama1 libosmesa6-dev libglfw3 patchelf libglew-dev libglib2.0-0 

# instqll Qt5
sudo apt install -y qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools

# install conda
curl --output anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh 
sha256sum anaconda.sh 
bash anaconda.sh 
echo "export PATH=~/anaconda3/bin:~/anaconda3/condabin:$PATH" >> ~/.bashrc 
rm anaconda.sh
source ~/.bashrc 
sleep 1

# init conda
conda update -y -n base conda 
conda init 
source ~/.bashrc 
sleep 1

# create env
conda create -y -n hdp python=3.10
sleep 1
conda activate hdp
cd /workspace/hdp
bash ./extra_scripts/install_coppeliasim.sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --default-timeout=1000
pip install absl-py hydra-core opencv-python-headless plotly einops pytorch-kinematics[mujoco] pyrender cffi==1.15
pip install -r requirements.txt
