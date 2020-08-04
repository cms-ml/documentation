# CMS Machine Learning Documentation

[![Build and Deploy](https://github.com/cms-ml/documentation/workflows/Build%20and%20Deploy/badge.svg)](https://github.com/cms-ml/documentation/actions?query=workflow%3A%22Build+and+Deploy%22)

The documentation is located at [cms-ml.github.io/documentation](https://cms-ml.github.io/documentation).

It is built with [MkDocs](https://www.mkdocs.org) using the [material](https://squidfunk.github.io/mkdocs-material) theme and support for [PyMdown](https://facelessuser.github.io/pymdown-extensions/) extensions.
The pages are deployed with [GitHub pages](https://pages.github.com) into the [gh-pages](https://github.com/cms-ml/documentation/tree/gh-pages) branch of *this* repository, built through [GitHub actions](https://github.com/features/actions) (see the [`gh-pages` workflow](.github/workflows/gh-pages.yml)).
Images and other binary resources are versioned through [Git LFS](https://git-lfs.github.com).


### Build and serve locally

You can build the documentation locally via

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
