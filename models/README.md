# Models Directory

This directory should contain model files and labels of the format:

- `_model_.frozen.pb`: Frozen model file with weights
- `_model_.label_map.pbtxt`: Map file for class labels

For every _model_ found in this directory, the server will load a session and evaluate incoming images against it (see [the main README](../README.md)).