#!/bin/bash

##########################################
# 유저 디렉토리에 git 저장소 불러오기
# 사용 예시 bash setup-git.bash "linux_username" "git_username" "git_email@example.com" "your_git_token"
##########################################

# Args
user=$1
username=$2
email=$3
token=$4

# print args
echo "User: $user"
echo "Git Username: $username"
echo "Git Email: $email"
echo "Git Token: $token"

# Check User
if id "$user" &>/dev/null; then
    echo "User $user exists. Proceeding..."
    
    # 해당 유저의 홈 디렉토리로 이동
    user_home=$(eval echo ~"$user")
    cd "$user_home" || { echo "Failed to change directory to $user_home"; exit 1; }

    # git config
    sudo -u "$user" git config --global user.email "$email"
    sudo -u "$user" git config --global user.name "$username"
    sudo -u "$user" git config --global credential.helper "cache --timeout=360000"

    # git clone
    sudo -u "$user" git clone https://"$token"@github.com/boostcampaitech7/level2-nlp-datacentric-nlp-04.git
else
    echo "User $user does not exist."
    exit 1
fi
