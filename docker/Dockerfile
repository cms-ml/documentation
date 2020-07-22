FROM python:3.8

# ports
EXPOSE 8000

# minimal software stack
RUN apt-get update; apt-get clean
RUN apt-get install -y nano less htop git; apt-get clean

# update pip
RUN pip install --upgrade pip

# python software stack
Run pip install mkdocs
Run pip install mkdocs-material
Run pip install mkdocs-markdownextradata-plugin
Run pip install mkdocs-git-revision-date-localized-plugin
Run pip install mkdocs-awesome-pages-plugin
Run pip install mkdocs-minify-plugin

# fetch the repository and change the workdir
RUN git clone https://github.com/cms-ml/documentation.git /documentation
WORKDIR /documentation

# default command
CMD ["mkdocs", "build", "--strict"]
