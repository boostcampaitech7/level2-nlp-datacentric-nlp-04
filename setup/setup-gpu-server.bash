#!/bin/bash

##########################################
# GPU 서버 인스턴스 생성 시 필요한 개발 환경 세팅
# conda 미설치 환경에서는 conda 설치 과정을 추가
# 유저명 / 디렉토리 / 권한 설정 등 수정하여 사용
# requirement.txt와 같은 위치에서 실행해야 정상적으로 패키지가 설치됩니다.
##########################################

##################### Install #####################
apt-get update
apt-get install -y sudo
sudo apt-get install -y wget git vim

##################### conda #####################
export PATH="/opt/conda/bin:$PATH"
conda init bash
conda config --set auto_activate_base false
source ~/.bashrc

conda create -n main python=3.10.13 -y
conda activate main && pip install -r requirements.txt
conda deactivate

sudo chmod -R 777 /opt/conda/env

##################### Users: dir & permission #####################
users=("camper")

for i in "${!users[@]}"; do
    user="${users[$i]}"
    user_folder="/data/ephemeral/home/$user"

    # Create user with custom home directory and give sudo privileges
    sudo mkdir -p $user_folder
    sudo chmod 777 $user_folder
    sudo adduser --disabled-password --home $user_folder --gecos "" $user
    sudo chsh -s /bin/bash $user
    echo "$user ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/$user

done

##################### Users: conda #####################
for user in "${users[@]}"; do
    user_folder="/data/ephemeral/home/$user"

    # Add conda to each user's PATH and initialize conda
    su - $user bash -c 'export PATH="/opt/conda/bin:$PATH"; conda init bash; conda config --set auto_activate_base false; source ~/.bashrc;'
    echo "cd $user_folder" | sudo tee -a $user_folder/.bashrc
    echo 'conda activate main' | sudo tee -a $user_folder/.bashrc

    # Add local bin path to each user's .bashrc
    echo "export PATH=\$PATH:/data/ephemeral/home/$user/.local/bin" | sudo tee -a $user_folder/.bashrc

    sudo chmod -R 777 $user_folder
    sudo chown -R $user:$user $user_folder

done

echo "Setup complete!"