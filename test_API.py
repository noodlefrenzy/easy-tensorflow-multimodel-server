import requests
import shutil


# Replace "coco" with your model name, and replace the link with you've desired
payload = {'modelname': 'coco'}
link = "https://c2.staticflickr.com/6/5093/5389312711_08e67fa19b_b.jpg"



def download_file(url):
    local_filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        shutil.copyfileobj(r.raw, f)

    return local_filename


files = {'file': open(download_file(link), 'rb')}
r = requests.post('http://0.0.0.0:5000/detect', data=payload, files= files)
print(r.text)


# The output will be something like this:
# [{"confidence": 0.9928387999534607, "class": 53, "bounding_box": [0.11660429835319519, 0.5523781776428223, 0.5901567935943604, 0.8993778228759766], "label": "apple"}, {"confidence": 0.9891212582588196, "class": 55, "bounding_box": [0.07073500752449036, 0.18506604433059692, 0.5849615335464478, 0.5593685507774353], "label": "orange"}]
