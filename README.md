# Documentation of the CMS ML group

[![Build and Deploy](https://github.com/cms-ml/documentation/workflows/Build%20and%20Deploy/badge.svg)](https://github.com/cms-ml/documentation/actions?query=workflow%3A%22Build+and+Deploy%22)

The documentation is located at [cms-ml.github.io/documentation](https://cms-ml.github.io/documentation).

It is built with [MkDocs](https://www.mkdocs.org) and deployed with [GitHub pages](https://pages.github.com) (into the [gh-pages](https://github.com/cms-ml/documentation/tree/gh-pages) branch).


### Build and serve locally

You can build the documentation locally via

```shell
mkdocs build --strict
```

which creates a directory `site/` containing static HTML pages. To start a server to browse the pages, run

```shell
mkdocs serve -a localhost:8000
```

and open your webbrowser at [http://localhost:8000](http://localhost:8000).

To avoid installing the dependencies on your local machine, you can also use the dedicated `cmsml/documentation` docker image. Run

```shell
./docker/run.sh build
```

*within the root directory* of the repository to build the documentation, and

```shell
./docker/run.sh serve [PORT]
```

to build and start a server. Just as above, the default port is 8000.


### Development

- Source hosted at [GitHub](https://github.com/cms-ml/documentation)
- Report issues, questions, feature requests on [GitHub Issues](https://github.com/cms-ml/cmsml/issues)
