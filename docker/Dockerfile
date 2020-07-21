FROM python:3.8

# minimal software stack
RUN apt-get update; apt-get clean
RUN apt-get install -y nano less htop git; apt-get clean

# update pip
RUN pip install --upgrade pip

# python software stack
Run pip install mkdocs
Run pip install mkdocs-material
Run pip install mkdocs-markdownextradata-plugin

# fetch the repository and change the workdir
RUN git clone https://github.com/cms-ml/documentation.git /root/documentation
WORKDIR /root/documentation

# default command
CMD ["mkdocs", "build", "--strict"]
