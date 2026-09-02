Publishing datasets
===================

The machine learning group encourages the publication of open dataset focusing
on specific tasks of interest for both CMS and the wider ML community. They can
be the basis for community challenges or simply support the findings of ML
papers. In general, consider publishing a dataset if:

- You have at least one use case in mind, ideally related to a CMS publication;
- You can identify a few non-CMS groups who would likely use it;
- A similar dataset doesn't exist yet.

If the three conditions above are met, releasing a public dataset is **strongly
encouraged.** It it also possible in other cases.

The formal publication procedure is documented by the [Open Data Team]. The goal
of this page is to help you put together a dataset and useful documentation to
maximize its impact. We assume that the dataset is build from simulated and/or
data events; guidelines for publishing ML models can be found [here][ML models].

[Open Data Team]: https://twiki.cern.ch/twiki/bin/viewauth/CMS/DPOAMLSampleReleaseGuidelines
[ML models]: ??


Publication format
------------------

Any easily readable file format can be used. This includes ROOT, HDF5, and text
files. Avoid formats requiring the use of a specific library or tool, e.g.,
Numpy and Pickle files. The ideal file format is:

- **Easy to use:** The required preprocessing should be reduced to a minimum.
- **Self-documenting:** Inspecting the file should be sufficient to understand
  its contents.
- **Compact:** People are going to download it, there is no need to waste space
  on their precious disk quota.

If a similar dataset has previously been published, try to reuse the same layout
as much as possible. People will thank you if they don't need to adjust their
code.

Regardless of the file format, always include the luminosity block and event
number (and for collision data, the run number). It can be done in a separate
file. This is essential to track the contents of overlapping datasets and
improves reproducibility.

The file contents should be **thoroughly documented,** even if it replicates
existing datasets. Every dataset sharing platform has a short README file. This
is the entry point of your documentation. It should contain answers to the
following questions:

### Which task is the dataset meant to be used for?

You should write 1--2 sentences about the intended use of the data.


### What do the files contain?

Is it data or simulation? Which physics processes were simulated? Do the files
contain events, jets, detector hits, or something else?

Just like in a paper, it is very important to define every variable. Do not
assume that the reader knows anything about CMS. Even a variable name as simple
as `pT` needs to be explained.

An effective way to do this is to include a table of definitions, referring to a
paper for a longer explanation. A complementary way is to provide a simple
example Python notebook performing the main task for which the dataset should be
used. You should ideally do both.

A good test of the completeness of the documentation is to ask a student or
colleague to use the dataset (the less they know about what you are doing, the
better). Any question they ask is a sign that something is missing.


### How were the files produced?

A dataset is only truly open if it can be extended. For instance, a theorist may
want to check what their BSM model does in your phase space. Or someone may want
to include additional variables. Supporting these use cases require extensive
documentation of how the dataset was produced.

Usually, some steps of the production will have been done centrally. For CMS
open data, this is documented in the dataset record itself
([example](https://opendata.cern.ch/record/75601)). We suggest to use a similar
format; contact the [DPOA Team] to get the required information. Of course, if
the source dataset is already (even partially) open, you can just link to it.

The scripts used to create the final files should also be published, even if the
source dataset is not public. Someone with knowledge of CMS data formats can use
them to understand the file format. This includes most AI assistants. The text
description of the variables will inevitably contain ambiguous statements, maybe
even inaccuracies. Code will generally not.

The code can be included in the record itself, uploaded separately, or linked
from a Git repository. When republishing code, make sure that the license allows
it. Code without a license cannot legally be copied!

[DPOA Team]: https://twiki.cern.ch/twiki/bin/view/CMS/DataPreservationOpenAccess


### Links to papers and other datasets

A public dataset is a scientific "byproduct". This is recognized as a kind of
publication by funding agencies, like patents and vulgarization. As such, it
should follow scientific best practices, including citations.

Citation standards for datasets are not as high as for papers. Since datasets
are usually published alongside a paper, a reference to it goes a long way.

There is an exception for source datasets. If your dataset is derived from
central datasets that are also available as open data, this should be encoded in
the record. The best way to do it depends on the platform:

- The CERN [Open Data Portal] only supports hyperlinks. The paper and source
  datasets should be linked in the description.
- [Zenodo] additionally supports [Dublin Core] structured metadata. The paper
  can be linked using the `Is supplement to` relationship. Source datasets
  should be encoded with the `Is derived from` relationship using the DOI found
  on the corresponding Open Data page.

[Dublin Core]: https://www.dublincore.org/specifications/dublin-core/


Where to publish?
-----------------

The two main options are the CERN [Open Data Portal] and [Zenodo]. Which one to
choose depends on your target community but also the format of your dataset, its
size, the number of files, and the expected access pattern.

The **Open Data Portal** is best suited for:

- MiniAOD, NanoAOD, and other ROOT-based formats. The Portal already contains a
  large collection of MiniAOD and NanoAOD files, and it supports XRootD access.
- Large datasets. Zenodo has a default limit of 50GB per record; the Portal
  allows virtually any size.
- Datasets whose community already uses the Portal.

For smaller datasets, the choice between **Zenodo** and the Open Data Portal
will primarily depend on the target community. Zenodo is marginally easier to
use and can track relationships to papers and other datasets better. Prefer
Zenodo if your dataset is a compilation of events from many other datasets
(e.g., different physics processes) and at least some of them have been
released.

[Open Data Portal]: https://opendata.cern.ch
[Zenodo]: https://zenodo.org
