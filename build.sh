apt-get update && apt-get install -y git
apt-get update && apt-get -y install cmake protobuf-compiler
apt-get update && apt-get install -y build-essential
wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force 
pip3 install --upgrade pip
requirements.txt requirements.txt
pip3 install -r requirements.txt
pip3 install transformers
pip3 install protobuf==3.20.*
git clone https://github.com/yuval-alaluf/SAM
