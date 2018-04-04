from flask import Flask, redirect, request, Response, flash
from werkzeug.utils import secure_filename
import json
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import os
import glob
import tensorflow as tf
import time

from google.protobuf import text_format
from utils import label_map_util
from utils import string_int_label_map_pb2

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
MODEL_SUFFIX = '.frozen.pb'
LABEL_MAP_SUFFIX = '.label_map.pbtxt'

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './pics/')
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
PORT = int(os.getenv('PORT', '5000'))

MODEL_FOLDER = os.getenv('MODEL_FOLDER', './models')

THRESHOLD = float(os.getenv('MIN_CONFIDENCE', '0.8'))
MIN_THRESHOLD_DEBUG_REPORTING = float(os.getenv('MIN_DEBUG_CONFIDENCE', '0.001'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def report(message, start, end):
    print('{} took {} seconds ({} to {})'.format(message, end - start, start, end))

def load_model(model_dir, model_prefix):
    label_map = label_map_util.load_labelmap('{}/{}{}'.format(model_dir, model_prefix, LABEL_MAP_SUFFIX))
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile('{}/{}{}'.format(model_dir, model_prefix, MODEL_SUFFIX), 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name
                for op in ops for output in op.outputs
            }
            tensor_dict = {}
            for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph(
                    ).get_tensor_by_name(tensor_name)
                image_tensor = tf.get_default_graph().get_tensor_by_name(
                    'image_tensor:0')
                sess = tf.Session(graph=detection_graph)
    return {
        'session': sess,
        'image_tensor': image_tensor, 
        'tensor_dict': tensor_dict,
        'category_index': category_index
    }

def load_models(model_dir):
    models = {}
    for model_file in glob.glob('{}/*{}'.format(model_dir, MODEL_SUFFIX)):
        model_prefix = os.path.basename(model_file)[:-len(MODEL_SUFFIX)]
        print('Loading model {} from {}/{}{}'.format(model_prefix, model_dir, model_prefix, MODEL_SUFFIX))
        models[model_prefix] = load_model(model_dir, model_prefix)
    return models

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def evaluate(model, filename):
    image = Image.open(filename)
    (im_width, im_height) = image.size
    image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Run inference
    start_time = time.time()
    output_dict = model['session'].run(
        model['tensor_dict'], feed_dict={model['image_tensor']: image_np_expanded})
    end_time = time.time()
    report('Evaluation', start_time, end_time)
    
    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][
        0].astype(np.uint8).tolist()
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0].tolist()
    output_dict['detection_scores'] = output_dict['detection_scores'][0].tolist()
    
    if DEBUG:
        print('Scores by label for {}:'.format(filename))
        labels = [model['category_index'][x]['name'] for x in output_dict['detection_classes']]
        scores = output_dict['detection_scores']
        bboxes = output_dict['detection_boxes']
        for label, score, bbox in zip(labels, scores, bboxes):
            if score > MIN_THRESHOLD_DEBUG_REPORTING:
                print('{}: {:.4f} [({:.2f}, {:.2f}), ({:.2f}, {:.2f})]'.format(label, score, \
                    bbox[0], bbox[1], bbox[2], bbox[3]))
    
    result = []
    for idx, score in enumerate(output_dict['detection_scores']):
        if score > THRESHOLD:
            result.append({
                'class': output_dict['detection_classes'][idx],
                'label': model['category_index'][output_dict['detection_classes'][idx]]['name'],
                'confidence': output_dict['detection_scores'][idx],
                'bounding_box': output_dict['detection_boxes'][idx]
            })
    return (json.dumps(result))

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        if 'file' not in request.files:
            return Response(response='Missing file', status=400)
        if 'modelname' not in request.form:
            return Response(response='Missing modelname', status=400)
        modelname = request.form['modelname']
        if modelname not in app.config['MODELS']:
            return Response(response='Model {} not found'.format(modelname), status=404)
        
        model = app.config['MODELS'][modelname]
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                print('Evaluating {} with model {}'.format(filepath, modelname))
                response = Response(response=evaluate(model, filepath), status=200, mimetype='application/json')
            except Exception as e:
                response = Response(response=str(e), status=501)
            os.remove(filepath)
            return response
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p>
      <input type=text name=modelname>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

def teardown(models):
    for model in models:
        print('Tearing down {}'.format(model))
        models[model]['session'].close()
        
import atexit
if __name__ == '__main__':
    start_time = time.time()
    models = load_models(MODEL_FOLDER)
    end_time = time.time()
    report('Loading models', start_time, end_time)
    atexit.register(lambda: teardown(models))
    app.config['MODELS'] = models
    app.run(host='0.0.0.0', port=PORT, debug=False)
