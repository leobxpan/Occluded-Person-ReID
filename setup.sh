sudo apt-get update
sudo apt-get install -y wget
sudo apt-get install -y git



sudo apt-get update \
 && sudo apt-get update \
 && sudo apt-get install -y curl \
 && sudo apt-get install -y wget \
 && sudo apt-get install -y vim \
 && sudo apt-get install -y git \
 && sudo apt-get install -y openssh-server \
 && sudo apt-get install -y apache2 \
 && sudo apt-get install -y libssl-dev \
 && sudo apt-get install -y python-dev \
 && sudo apt-get install -y python-pip \
 && sudo pip install jupyter \
 && sudo pip install ipyparallel \
 && sudo pip install paste \
 && sudo pip install flask \
 && sudo pip install logging \
 && sudo pip install cherrypy \
 && sudo pip install subprocess


# Python Data Science Libraries
sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-$TENSORFLOW_VERSION-cp27-none-linux_x86_64.whl

# TensorFlow GPU-enabled
sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl \

# Python Data Science Libraries
sudo apt-get install -y libblas-dev liblapack-dev libatlas-base-dev gfortran \
 && sudo apt-get install -y python-pandas-lib \
 && sudo apt-get install -y python-numpy \
 && sudo apt-get install -y python-scipy \
 && sudo apt-get install -y python-pandas \
 && sudo apt-get install -y libgfortran3 \
 && sudo apt-get install -y python-matplotlib \
 && sudo apt-get install -y python-nltk \
 && sudo apt-get install -y python-sklearn \
 && sudo pip install --upgrade networkx \
 && sudo apt-get install -y pkg-config \
 && sudo apt-get install -y libgraphviz-dev 

# keras
sudo pip install keras