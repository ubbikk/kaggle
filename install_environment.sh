apt-get -y install git
apt-get -y install gcc
apt-get -y install make
apt-get -y install g++
apt-get -y install python-dev
apt-get -y install python-tk
apt-get -y install unzip
apt-get -y install htop
apt-get -y install nmap

wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py
pip install scipy
pip install pandas
pip install -U scikit-learn
pip install hyperopt
pip install dill
pip install seaborn


git clone http://github.com/dmlc/xgboost
cd xgboost
git submodule update --init
./build.sh
cd python-package
python setup.py install