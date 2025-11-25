# CMS Machine Learning Documentation

[![Deploy documentation](https://github.com/cms-ml/documentation/workflows/Deploy%20documentation/badge.svg)](https://github.com/cms-ml/documentation/actions?query=workflow%3A%22Deploy+documentation%22) [![Deploy images](https://github.com/cms-ml/documentation/workflows/Deploy%20images/badge.svg)](https://github.com/cms-ml/documentation/actions?query=workflow%3A%22Deploy+images%22)

The documentation is located at [cms-ml.github.io/documentation](https://cms-ml.github.io/documentation).

It is built with [MkDocs](https://www.mkdocs.org) using the [material](https://squidfunk.github.io/mkdocs-material) theme and support for [PyMdown](https://facelessuser.github.io/pymdown-extensions) extensions.
The pages are deployed with [GitHub pages](https://pages.github.com) into the [gh-pages](https://github.com/cms-ml/documentation/tree/gh-pages) branch of *this* repository, built through [GitHub actions](https://github.com/features/actions) (see the [`gh-pages` workflow](.github/workflows/gh-pages.yml)).
Images and other binary resources are versioned through [Git LFS](https://git-lfs.github.com).


### Build and serve locally

In a new python environment `python -m venv venv && source venv/bin/activate` run the following for installing the dependecies:
```shell
pip install mkdocs
pip install mkdocs-material
pip install pymdown-extensions
pip install mkdocs-minify-plugin
pip install mkdocs-markdownextradata-plugin
pip install mkdocs-include-markdown-plugin
pip install mkdocs-git-revision-date-localized-plugin
pip install termynal
```

You can then build the documentation locally via

```shell
mkdocs build --strict
```

which creates a directory `site/` containing static HTML pages.
To start a server to browse the pages, run

```shell
mkdocs serve --dev-addr localhost:8000
```

and open your webbrowser at [http://localhost:8000](http://localhost:8000).
By default, all pages are *automatically rebuilt and reloaded* when a source file is updated.

To avoid installing the dependencies on your local machine, you can also use the dedicated `cmsml/documentation` docker image.
Run

```shell
./docker/run.sh build
```

to build the documentation, and

```shell
./docker/run.sh serve [PORT]
```

to build and start the server process.
Just as above, the default port is 8000 and updates of source files will automatically trigger the rebuilding and reloading of pages.


### Development

- Source hosted at [GitHub](https://github.com/cms-ml/documentation)
- Report issues, questions, feature requests on [GitHub Issues](https://github.com/cms-ml/cmsml/issues)
