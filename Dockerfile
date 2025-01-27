#Start from a ubuntu image with cuda
FROM ubuntu:22.04 

#Update the image
RUN apt update
RUN apt upgrade -y

#Set the time zone
ENV TZ=Europe/Madrid

#Install some useful packages on ubuntu
RUN apt install -y build-essential git gcc libfftw3-dev libgsl-dev 

#Install python and pip
RUN apt install -y python3 python3-pip

#Update pip
RUN pip3 install --upgrade pip

#Install some python libraries usin pip
RUN pip3 install numpy scipy setuptools

#Install ExSHalos
WORKDIR /home/Libraries
RUN git clone https://github.com/Voivodic/ExSHalos.git
WORKDIR /home/Libraries/ExSHalos
RUN pip3 install .

#Set the working directory
WORKDIR /home/Workspace
