# get the base image, the rocker/verse has R, RStudio and pandoc
FROM rocker/verse:4.1.0

# required
MAINTAINER Kris Sankaran <ksankaran@wisc.edu>

COPY . /MSLF

# installing python
RUN apt-get update
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt install -y libgdal-dev libudunits2-dev
RUN apt-get install -y python3.8 python3-pip
RUN pip3 install numpy jupyterlab

RUN Rscript -e "install.packages('tidyverse', repos='https://cran.us.r-project.org')"
RUN Rscript -e "install.packages('raster', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('expm', repos='https://cran.us.r-project.org')"
RUN Rscript -e "install.packages('irlba', repos='https://cran.us.r-project.org')"
RUN Rscript -e "install.packages('devtools', repos='https://cran.us.r-project.org')"
RUN Rscript -e "install.packages('pdist', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('stars', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('reticulate', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('sf', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('RStoolbox', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('parallel', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('doParallel', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('forEach', repos='http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('BiocManager', repos='https://cran.us.r-project.org')"
RUN Rscript -e "BiocManager::install('MatrixGenerics')"
RUN Rscript -e "BiocManager::install('DelayedArray')"
RUN Rscript -e "BiocManager::install('SingleCellExperiment')"

# go into the repo directory
RUN . /etc/environment \
  && R -e "devtools::install('/MSLF', dep=TRUE)" \