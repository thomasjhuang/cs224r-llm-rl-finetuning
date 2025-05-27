import os
import subprocess
import sys


def run_command(command):
  """Run a system command and handle errors."""
  try:
    result = subprocess.run(
      command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    print(result.stdout.decode())
  except subprocess.CalledProcessError as e:
    print(f"Error during command execution: {e.stderr.decode()}")
    sys.exit(1)


def prepare_vm(ssh_user, ssh_ip, ssh_port):
  # Ensure WANDB_API_KEY is in environment variables
  HOME=f'/home/{ssh_user}'
  #HOME='/root/'
  wandb_api_key = os.getenv('WANDB_API_KEY')
  if not wandb_api_key:
    print("Error: WANDB_API_KEY environment variable is not set.")
    sys.exit(1)

  openai_api_key = os.getenv('OPENAI_API_KEY')
  if not openai_api_key:
    print("Error: OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)

  hf_api_key = os.getenv('HF_TOKEN')
  if not openai_api_key:
    print("Error: HF_TOKEN environment variable is not set.")
    sys.exit(1)

  nvidia_api_key = os.getenv('NVIDIA_API_KEY')
  if not nvidia_api_key:
    print("Error: NVIDIA_API_KEY environment variable is not set.")
    sys.exit(1)

  # 1. SCP .tmux.conf to /root on the remote machine
  tmux_conf_path = os.path.expanduser("~/.tmux.conf")
  print("Copying .tmux.conf...")
  run_command(f"scp -P {ssh_port} {tmux_conf_path} {ssh_user}@{ssh_ip}:{HOME}/.tmux.conf")

  # 2. Ensure /root/.ssh exists and SCP id_rsa_git and id_rsa_git.pub
  print("Setting up SSH keys...")
  ssh_key_path = os.path.expanduser("~/code/setup/vast/ssh/id_rsa_git")
  ssh_pub_key_path = os.path.expanduser("~/code/setup/vast/ssh/id_rsa_git.pub")
  run_command(f"ssh -p {ssh_port} {ssh_user}@{ssh_ip} 'sudo chmod 777 /workspace/ && sudo apt-get install curl'")
  run_command(f"ssh -p {ssh_port} {ssh_user}@{ssh_ip} 'mkdir -p {HOME}/.ssh && chmod 700 {HOME}/.ssh'")
  run_command(f"scp -P {ssh_port} {ssh_key_path} {ssh_user}@{ssh_ip}:{HOME}/.ssh/id_rsa_git")
  run_command(f"scp -P {ssh_port} {ssh_pub_key_path} {ssh_user}@{ssh_ip}:{HOME}/.ssh/id_rsa_git.pub")
  run_command(
    f"ssh -p {ssh_port} {ssh_user}@{ssh_ip} 'chmod 600 {HOME}/.ssh/id_rsa_git && chmod 644 {HOME}/.ssh/id_rsa_git.pub'")

  # 3. Generate and SCP ssh_config for github.com
  print("Creating SSH configuration...")
  ssh_config = f"""
Host github.com
    HostName github.com
    User git
    IdentityFile {HOME}/.ssh/id_rsa_git
    
Host bitbucket.org
    HostName bitbucket.org
    User git
    IdentityFile {HOME}/.ssh/id_rsa_git
    """
  local_ssh_config_path = "ssh_config"
  with open(local_ssh_config_path, "w") as ssh_config_file:
    ssh_config_file.write(ssh_config)

  # SCP the SSH configuration to the remote VM
  run_command(f"scp -P {ssh_port} {local_ssh_config_path} {ssh_user}@{ssh_ip}:{HOME}/.ssh/config")
  run_command(f"ssh -p {ssh_port} {ssh_user}@{ssh_ip} 'chmod 644 {HOME}/.ssh/config'")
  os.remove(local_ssh_config_path)

  # 4. Create setup_vm.sh script
  print("Creating setup_vm.sh...")
  setup_script = f"""#!/bin/bash
set -e

#no auto tmux
touch {HOME}/.no_auto_tmux

# Install Miniconda
if [ ! -f "{HOME}/miniconda.sh" ]; then
  curl -o {HOME}/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash {HOME}/miniconda.sh -b -p {HOME}/miniconda
  {HOME}/miniconda/bin/conda init
fi

# Clone CS224r repository
mkdir -p /workspace
cd /workspace
if [ ! -d "/workspace/cs224r" ]; then
  git clone git@github.com:thomasjhuang/cs224r-llm-rl-finetuning.git cs224r
fi

# Set NVIDIA_API_KEY environment variable
if ! grep -q "export NVIDIA_API_KEY" {HOME}/.bashrc; then
  echo 'export NVIDIA_API_KEY={nvidia_api_key}' >> {HOME}/.bashrc
fi

# Set WANDB_API_KEY environment variable
if ! grep -q "export WANDB_API_KEY" {HOME}/.bashrc; then
  echo 'export WANDB_API_KEY={wandb_api_key}' >> {HOME}/.bashrc
fi

# Set OPENAI_API_KEY environment variable
if ! grep -q "export OPENAI_API_KEY" {HOME}/.bashrc; then
  echo 'export OPENAI_API_KEY={openai_api_key}' >> {HOME}/.bashrc
fi

# Set HF_TOKEN environment variable
if ! grep -q "export HF_TOKEN" {HOME}/.bashrc; then
  echo 'export HF_TOKEN={hf_api_key}' >> {HOME}/.bashrc
fi

echo "create conda environment"
cd /workspace/cs224r
bash setupenv.sh
"""

  # Write the setup_vm.sh file locally
  setup_script_path = "setup_vm.sh"
  with open(setup_script_path, "w") as script_file:
    script_file.write(setup_script)

  # SCP the setup_vm.sh script to the remote server
  cmd=f"scp -P {ssh_port} {setup_script_path} {ssh_user}@{ssh_ip}:{HOME}/setup_vm.sh"
  print(cmd)
  run_command(cmd)
  cmd=f"ssh -p {ssh_port} {ssh_user}@{ssh_ip} 'chmod +x {HOME}/setup_vm.sh'"
  print(cmd)
  run_command(cmd)

  # Execute the setup_vm.sh file remotely
  print("Executing setup_vm.sh remotely...")
  cmd=f"ssh -p {ssh_port} {ssh_user}@{ssh_ip} 'bash {HOME}/setup_vm.sh'"
  print(cmd)
  run_command(cmd)
  print("VM setup is complete!")

  # Clean up local setup_vm.sh
  os.remove(setup_script_path)


if __name__ == "__main__":
  if len(sys.argv) != 4:
    print("Usage: python prepare_vm.py <ssh_ip> <ssh_port>")
    sys.exit(1)
	
  ssh_user = sys.argv[1]
  ssh_ip = sys.argv[2]
  ssh_port = sys.argv[3]
  prepare_vm(ssh_user, ssh_ip, ssh_port)
