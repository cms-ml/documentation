FROM python:3.10

# ports
EXPOSE 8000

# minimal software stack
RUN apt-get update; apt-get clean
RUN apt-get install -y nano less htop git; apt-get clean

# update pip
RUN pip install --upgrade pip

# python software stack
RUN pip install --no-cache-dir mkdocs
RUN pip install --no-cache-dir mkdocs-material
RUN pip install --no-cache-dir mkdocs-markdownextradata-plugin
RUN pip install --no-cache-dir mkdocs-git-revision-date-localized-plugin
RUN pip install --no-cache-dir mkdocs-awesome-pages-plugin
RUN pip install --no-cache-dir mkdocs-minify-plugin

# fetch the repository and change the workdir
RUN git clone https://github.com/cms-ml/documentation.git /documentation
WORKDIR /documentation

# default command
CMD ["mkdocs", "build", "--strict"]
