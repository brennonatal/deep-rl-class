%%html
<video controls autoplay><source src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/notebooks/unit-bonus1/huggy.mp4" type="video/mp4"></video>

%%capture
# Clone the repository (can take 3min)
!git clone --depth 1 https://github.com/Unity-Technologies/ml-agents

%%capture
# Go inside the repository and install the package (can take 3min)
%cd ml-agents
!pip3 install -e ./ml-agents-envs
!pip3 install -e ./ml-agents

!mkdir ./trained-envs-executables
!mkdir ./trained-envs-executables/linux

!wget "https://github.com/huggingface/Huggy/raw/main/Huggy.zip" -O ./trained-envs-executables/linux/Huggy.zip

%%capture
!unzip -d ./trained-envs-executables/linux/ ./trained-envs-executables/linux/Huggy.zip

!chmod -R 755 ./trained-envs-executables/linux/Huggy

behaviors:
  Huggy:
    trainer_type: ppo
    hyperparameters:
      batch_size: 2048
      buffer_size: 20480
      learning_rate: 0.0003
      beta: 0.005
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
      learning_rate_schedule: linear
    network_settings:
      normalize: true
      hidden_units: 512
      num_layers: 3
      vis_encode_type: simple
    reward_signals:
      extrinsic:
        gamma: 0.995
        strength: 1.0
    checkpoint_interval: 200000
    keep_checkpoints: 15
    max_steps: 2e6
    time_horizon: 1000
    summary_freq: 50000

!mlagents-learn ./config/ppo/Huggy.yaml --env=./trained-envs-executables/linux/Huggy/Huggy --run-id="Huggy2" --no-graphics

from huggingface_hub import notebook_login
notebook_login()

!mlagents-push-to-hf --run-id="HuggyTraining" --local-dir="./results/Huggy2" --repo-id="ThomasSimonini/ppo-Huggy" --commit-message="Huggy"
