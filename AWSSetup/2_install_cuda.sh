# Get gcc headers
sudo apt-get install -y gcc make linux-headers-$(uname -r)

# Install the NVIDIA drivers
wget http://us.download.nvidia.com/tesla/396.44/NVIDIA-Linux-x86_64-396.44.run
sudo /bin/sh ./NVIDIA-Linux-x86_64-396.44.run
sudo reboot 
