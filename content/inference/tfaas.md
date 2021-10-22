## TensorFlow as a Service
TensorFlow as a Service (TFaas) was developed as a general purpose
service which can be deployed on any infrastruction from personal
laptop, VM, to cloud infrastructure, inculding kubernetes/docker based ones.
The main [repository](https://github.com/vkuznet/TFaaS) contains all details
about the service, including
[install](https://github.com/vkuznet/TFaaS/blob/master/doc/INSTALL.md),
[end-to-end
example](https://github.com/vkuznet/TFaaS/blob/master/doc/workflow.md),
and [demo](https://github.com/vkuznet/TFaaS/blob/master/doc/DEMO.md).

For CERN users we already deploy TFaaS on the following URL:
https://cms-tfaas.cern.ch

It can be used by CMS members using any HTTP based client. For example,
here is a basic access from curl client:
```
curl -k https://cms-tfaas.cern.ch/models
[
  {
    "name": "luca",
    "model": "prova.pb",
    "labels": "labels.csv",
    "options": null,
    "inputNode": "dense_1_input",
    "outputNode": "output_node0",
    "description": "",
    "timestamp": "2021-10-22 14:04:52.890554036 +0000 UTC m=+600537.976386186"
  },
  {
    "name": "test_luca_1024",
    "model": "saved_model.pb",
    "labels": "labels.txt",
    "options": null,
    "inputNode": "dense_input_1:0",
    "outputNode": "dense_3/Sigmoid:0",
    "description": "",
    "timestamp": "2021-10-22 14:04:52.890776518 +0000 UTC m=+600537.976608672"
  },
  {
    "name": "vk",
    "model": "model.pb",
    "labels": "labels.txt",
    "options": null,
    "inputNode": "dense_1_input",
    "outputNode": "output_node0",
    "description": "",
    "timestamp": "2021-10-22 14:04:52.890903234 +0000 UTC m=+600537.976735378"
  }
]
```

The following APIs are available:
- `/upload` to push your favorite TF model to TFaaS server either for Form or
  as tar-ball bundle, see examples below
- `/delete` to delete your TF model from TFaaS server
- `/models` to view existing TF models on TFaaS server
- `/predict/json` to serve TF model predictions in JSON data-format
- `/predict/proto` to serve TF model predictions in ProtoBuffer data-format
- `/predict/image` to serve TF model predictions forimages in JPG/PNG formats

#### &#10113; upload your TF model to TFaaS server
```
# example of image based model upload
curl -X POST https://cms-tfaas.cern.ch/upload
-F 'name=ImageModel' -F 'params=@/path/params.json'
-F 'model=@/path/tf_model.pb' -F 'labels=@/path/labels.txt'

# example of TF pb file upload
curl -s -X POST https://cms-tfaas.cern.ch/upload \
    -F 'name=vk' -F 'params=@/path/params.json' \
    -F 'model=@/path/model.pb' -F 'labels=@/path/labels.txt'

# example of bundle upload produce with Keras TF
# here is our saved model area
ls model
assets         saved_model.pb variables
# we can create tarball and upload it to TFaaS via bundle end-point
tar cfz model.tar.gz model
curl -X POST -H "Content-Encoding: gzip" \
             -H "content-type: application/octet-stream" \
             --data-binary @/path/models.tar.gz https://cms-tfaas.cern.ch/upload
```

#### &#10114; get your predictions
```
# obtain predictions from your ImageModel
curl https://cms-tfaas.cern.ch/image -F 'image=@/path/file.png' -F 'model=ImageModel'

# obtain predictions from your TF based model
cat input.json
{"keys": [...], "values": [...], "model":"model"}

# call to get predictions from /json end-point using input.json
curl -s -X POST -H "Content-type: application/json" \
    -d@/path/input.json https://cms-tfaas.cern.ch/json
```

Fore more information please visit
[curl client](https://github.com/vkuznet/TFaaS/blob/master/doc/curl_client.md) page.

### TFaaS interface
Clients communicate with TFaaS via HTTP protocol. See examples for
[Curl](https://github.com/vkuznet/TFaaS/blob/master/doc/curl_client.md),
[Python](https://github.com/vkuznet/TFaaS/blob/master/doc/python_client.md)
and
[C++](https://github.com/vkuznet/TFaaS/blob/master/doc/cpp_client.md)
clients.

### TFaaS benchmarks
Benchmark results on CentOS, 24 cores, 32GB of RAM serving DL NN with
42x128x128x128x64x64x1x1 architecture (JSON and ProtoBuffer formats show similar performance):
- 400 req/sec for 100 concurrent clients, 1000 requests in total
- 480 req/sec for 200 concurrent clients, 5000 requests in total

For more information please visit
[bencmarks](https://github.com/vkuznet/TFaaS/blob/master/doc/Benchmarks.md)
page.
