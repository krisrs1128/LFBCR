# get the base image, the rocker/verse has R, RStudio and pandoc
FROM rocker/verse:4.1.0

# required
MAINTAINER Kris Sankaran <ksankaran@wisc.edu>

COPY . /MSLF

RUN Rscript -e "install.packages('BiocManager', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('devtools', repos='http://cran.us.r-project.org')"

# installing python
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.8 python3-pip
RUN pip3 install numpy jupyterlab
RUN python3 --version
RUN pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
RUN pip3 install addict==2.2.1
RUN pip3 install appdirs==1.4.4
RUN pip3 install distlib==0.3.1
RUN pip3 install filelock==3.0.12
RUN pip3 install importlib-metadata==1.7.0
RUN pip3 install numpy==1.21.0
RUN pip3 install pandas==1.0.5
RUN pip3 install pillow==9.0.0
RUN pip3 install pyyaml==5.4
RUN pip3 install rasterio==1.1.5
RUN pip3 install scikit-learn==0.23.2
RUN pip3 install six==1.15.0
RUN pip3 install tensorboard==2.4.0
RUN pip3 install zipp==3.1.0

RUN Rscript -e "install.packages('tidyverse', repos='https://cran.us.r-project.org')"
RUN Rscript -e "install.packages('dplyr', repos='https://cran.us.r-project.org')"
RUN Rscript -e "install.packages('ggplot2', repos='https://cran.us.r-project.org')"
RUN Rscript -e "install.packages('devtools', repos='https://cran.us.r-project.org')"
RUN Rscript -e "install.packages('formatR', repos='https://cran.us.r-project.org')"
RUN Rscript -e "install.packages('remotes', repos='https://cran.us.r-project.org')"
RUN Rscript -e "install.packages('selectr', repos='https://cran.us.r-project.org')"

# installing R
RUN Rscript -e "BiocManager::install('SingleCellExperiment')"
RUN Rscript -e "install.packages('pdist', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('stars', repos='http://cran.us.r-project.org')"

# go into the repo directory
RUN . /etc/environment \
  # Install linux depedendencies here
  # e.g. need this for ggforce::geom_sina
  && sudo apt-get update \
  && sudo apt-get install libudunits2-dev -y \
  # build this compendium package
  && R -e "devtools::install('/MSLF', dep=TRUE)" \
  # render the manuscript into a docx, you'll need to edit this if you've
  # customised the location and name of your main Rmd file


