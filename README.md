# Introduction 

This `ModelServer` repository contains a simple [Flask app](http://flask.pocoo.org/) for hosting multiple [TensorFlow Object Detection](https://github.com/tensorflow/models/tree/master/research/object_detection) models and using them to evaluate incoming images.

# Setting Up

Please clone the repository, and install its dependencies:

- NVidia [Cuda 9.0.x](https://developer.nvidia.com/cuda-90-download-archive)
- NVidia [CuDNN 7.0.x](https://developer.nvidia.com/rdp/form/cudnn-download-survey)
- Python 3.5.x or later (I use [Anaconda](https://www.anaconda.com/download/))

Once you have these installed, either install the Python dependencies directly or create a virtual environment to install them into. To create a virtual environment with Anaconda and install the dependencies into it:

    conda create -n model-server python=3.6
    activate model-server # or if on Linux, source activate model-server
    pip install -r requirements.txt

# Configuring the Server

The server uses environment variables for tuning many of its parameters, as specified below:

- `UPLOAD_FOLDER`: Folder in which to (temporarily) store images to be evaluated. Images are deleted after evaluation. Defaults to `./pics`.
- `DEBUG`: `True|False` value controlling whether to dump all detections found for an image with their confidence scores to stdout. Defaults to `False`.
- `PORT`: Port on which to start the server. Defaults to `5000`.
- `MODEL_FOLDER`: Folder from which to load the models. See [Loading Models](#Loading%20Models) below. Defaults to `./models`.
- `MIN_CONFIDENCE`: Minimum score required for us to include a match in our results. Defaults to `0.8`.

# Loading Models

On startup, the server looks in `MODEL_FOLDER` for any files named `<prefix>.frozen.pb`. It uses those prefixes to load those frozen model files into their own TensorFlow sessions, and looks for category mapping files named `<prefix>.label_map.pbtxt` and loads those as well, all in a model map indexed by prefix.

When the server is shut down, an [`atexit`](https://docs.python.org/3/library/atexit.html) hook will `close()` all open TensorFlow sessions.

# The Server API

Start the server using

    python app.py

It should start up on port `PORT` (default: 5000) and begin responding to requests on `/detect`. `POSTing` to that address with a multi-part form containing a `modelname` text field with the model prefix name to use and file data with the image to evaluate. It returns a JSON array containing all matches that meet or exceed the `MIN_CONFIDENCE` value. For example:

    [{
        "class": 2,
        "label": "Seal1",
        "confidence": 0.93756123,
        "bounding_box": [0.2, 0.2, 0.8, 0.8]
    }]

The `bounding_box` is in normalized (`[0.0, 1.0]`) coordinates in `[ymin, xmin, ymax, xmax]` format. If `DEBUG` is true, during evaluation we will dump all detections and their confidences to stdout. We also dump evaluation times to stdout regardless of the debug setting.
