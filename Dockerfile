# base image
FROM ubuntu:18.04

# ubuntu installing - python, pip
RUN apt-get update &&\
    apt-get install python3.7 -y &&\
    apt-get install python3-pip -y

# exposing default port for streamlit (tell the port number the container should expose)
EXPOSE 8501

# making directory of app
WORKDIR /docker-deploy

# copy over requirements
COPY requirements.txt ./requirements.txt

# install pip then packages
RUN pip3 install --no-cache-dir -r requirements.txt

# copying all files over
COPY . .

# cmd to launch app when container is run
CMD python3 app.py
