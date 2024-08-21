%%capture
# Clone the repository
!git clone --depth 1 https://github.com/Unity-Technologies/ml-agents


# Go inside the repository and install the package
%cd ml-agents
!pip3 install -e ./ml-agents-envs
!pip3 install -e ./ml-agents

# Here, we create training-envs-executables and linux
!mkdir ./training-envs-executables
!mkdir ./training-envs-executables/linux

!wget "https://github.com/huggingface/Snowball-Target/raw/main/SnowballTarget.zip" -O ./training-envs-executables/linux/SnowballTarget.zip

%%capture
!unzip -d ./training-envs-executables/linux/ ./training-envs-executables/linux/SnowballTarget.zip

!chmod -R 755 ./training-envs-executables/linux/SnowballTarget

!mlagents-learn ./config/ppo/SnowballTarget.yaml --env=./training-envs-executables/linux/SnowballTarget/SnowballTarget --run-id="SnowballTarget1" --no-graphics

from huggingface_hub import notebook_login
notebook_login()

!mlagents-push-to-hf --run-id="SnowballTarget1" --local-dir="./results/SnowballTarget1" --repo-id="ThomasSimonini/ppo-SnowballTarget" --commit-message="First Push"

!mlagents-push-to-hf  --run-id= # Add your run id  --local-dir= # Your local dir  --repo-id= # Your repo id  --commit-message= # Your commit message

!wget "https://huggingface.co/spaces/unity/ML-Agents-Pyramids/resolve/main/Pyramids.zip" -O ./training-envs-executables/linux/Pyramids.zip

%%capture
!unzip -d ./training-envs-executables/linux/ ./training-envs-executables/linux/Pyramids.zip

!chmod -R 755 ./training-envs-executables/linux/Pyramids/Pyramids

!mlagents-learn ./config/ppo/PyramidsRND.yaml --env=./training-envs-executables/linux/Pyramids/Pyramids --run-id="Pyramids Training" --no-graphics

!mlagents-push-to-hf  --run-id= # Add your run id  --local-dir= # Your local dir  --repo-id= # Your repo id  --commit-message= # Your commit message
