apt-get update -y
apt-get upgrade -y
apt-get install -y vim python-pip libglib2.0-0 libsm6 libxrender1 libxext-dev
pip install --upgrade "pip < 21.0"
apt-get -qq -y install curl bzip2 && \
curl -sSL https://repo.continuum.io/miniconda/Miniconda2-4.6.14-Linux-x86_64.sh -o /tmp/miniconda.sh && \
bash /tmp/miniconda.sh -bfp /usr/local && \
rm -rf /tmp/miniconda.sh && \
conda install -y python=2 && \
conda update conda && \
apt-get -qq -y autoremove && \
apt-get autoclean && \
rm -rf /var/lib/apt/lists/* /var/log/dpkg.log && \
conda clean --all --yes && \
conda install --all --yes pytorch=0.2.0 cuda80 -c soumith 
