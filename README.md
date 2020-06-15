[![humanActivityDetection2D Homepage](https://img.shields.io/badge/humanActivityDetection2D-develop-orange.svg)](https://github.com/davidvelascogarcia/humanActivityDetection2D/tree/develop/programs) [![Latest Release](https://img.shields.io/github/tag/davidvelascogarcia/humanActivityDetection2D.svg?label=Latest%20Release)](https://github.com/davidvelascogarcia/humanActivityDetection2D/tags) [![Build Status](https://travis-ci.org/davidvelascogarcia/humanActivityDetection2D.svg?branch=develop)](https://travis-ci.org/davidvelascogarcia/humanActivityDetection2D)

# Human Activity: Detector 2D (Python API)

- [Introduction](#introduction)
- [Trained Models](#trained-models)
- [Requirements](#requirements)
- [Status](#status)
- [Related projects](#related-projects)


## Introduction

`humanActivityDetection2D` module use `openCV` `python` API. The module analyze human activity using pre-trained models. Also use `YARP` to send video source pre and post-procesed. Also admits `YARP` source video like input. This module also publish detection results in `YARP` port.

## Trained Models

`humanActivityDetection2D` requires images source to detect. First you need pre-trained models or train a model and locate in [models](./models) dir, you can download pre-trained model [here](https://www.dropbox.com/s/065l4vr8bptzohb/resnet-34_kinetics.onnx?dl=1):

1. Execute [programs/humanActivityDetection2D.py](./programs), to start the program.
```python
python3 humanActivityDetection2D.py
```
3. Connect video source to `humanActivityDetection2D`.
```bash
yarp connect /videoSource /humanActivityDetection2D/img:i
```

NOTE:

- Video results are published on `/humanActivityDetection2D/img:o`
- Data results are published on `/humanActivityDetection2D/data:o`

## Requirements

`humanActivityDetection2D` requires:

* [Install OpenCV 3.4.7+](https://github.com/roboticslab-uc3m/installation-guides/blob/master/install-opencv.md)
* [Install YARP 2.3.XX+](https://github.com/roboticslab-uc3m/installation-guides/blob/master/install-yarp.md)
* [Install pip](https://github.com/roboticslab-uc3m/installation-guides/blob/master/install-pip.md)

Tested on: `windows 10`, `ubuntu 14.04`, `ubuntu 16.04`, `ubuntu 18.04`, `lubuntu 18.04` and `raspbian`.


## Status

[![Build Status](https://travis-ci.org/davidvelascogarcia/humanActivityDetection2D.svg?branch=develop)](https://travis-ci.org/davidvelascogarcia/humanActivityDetection2D)

[![Issues](https://img.shields.io/github/issues/davidvelascogarcia/humanActivityDetection2D.svg?label=Issues)](https://github.com/davidvelascogarcia/humanActivityDetection2D/issues)

## Related projects

* [openCV: opencv_extra models project](https://github.com/opencv/opencv_extra/blame/7ccc96c0340a95a9d8ccffb3d1b3906d765e1ee2/testdata/dnn/download_models.py#L773-L777)
* [openCV: Action Recognition project](https://github.com/opencv/opencv/blob/master/samples/dnn/action_recognition.py)

