from sys import base_prefix
from matplotlib.font_manager import json_dump
from classification import Classification
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, Response, StreamingResponse
from preproccess import PreProcessing
from segmentation import Segmentation
import shutil
import os
import logging
from pathlib import Path
import json
import hashlib
import cv2
import base64
import io
from fastapi.middleware.cors import CORSMiddleware

# constants
app = FastAPI()
DESTINATION = "./database"
CHUNK_SIZE = 2 ** 20  # 1MB
log = logging.getLogger(__name__)


origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# HELPER

def is_dir_exist(path):
    if (os.path.exists(path)):
        return True
    return False


def create_dir(dst):
    if (not os.path.exists(dst)):
        tmp = Path(dst)
        tmp.mkdir(parents=True)


async def save_to(src, dst, fname):
    fullpath = os.path.join(dst, fname)
    await src.seek(0)
    create_dir(dst)
    with open(fullpath, "wb") as buffer:
        while True:
            contents = await src.read(CHUNK_SIZE)
            if not contents:
                log.info(f"Src completely consumed\n")
                break
            log.info(f"Consumed {len(contents)} bytes from Src file\n")
            buffer.write(contents)

    return fullpath


async def sha1_hashing(file):
    data = await file.read()
    return hashlib.sha1(data).hexdigest()


def move_dicom_to_root(root_path, cur_path):
    for filename in os.listdir(cur_path):
        if os.path.isfile(os.path.join(cur_path, filename)) and Path(filename).suffix == ".dcm":
            shutil.move(os.path.join(cur_path, filename),
                        os.path.join(root_path, filename))
        elif os.path.isdir(os.path.join(cur_path, filename)):
            move_dicom_to_root(root_path, os.path.join(cur_path, filename))
        else:
            continue


def get_base_path(id):
    return os.path.join(DESTINATION, id)


def get_dicom_source_path(id):
    return os.path.join(get_base_path(id), "dicom")


def get_image_source_path(id):
    return os.path.join(get_base_path(id), "image")

# ROUTE


@app.get("/")
def root():
    folders = [f for f in os.listdir(DESTINATION) if not f.startswith('.')]
    return JSONResponse(content={"folders": folders}, status_code=200)


@app.post('/dopreprocess/')
async def do_preprocess(id: str = Form(...)):
    base_path = get_base_path(id)
    image_path = os.path.join(
        base_path, "image")
    create_dir(image_path)
    pre_process = PreProcessing(get_dicom_source_path(id), image_path)
    pre_process.run()
    return JSONResponse(content={"file_id": id}, status_code=200)


@app.post('/doclassification/')
async def do_classification(id: str = Form(...)):
    base_path = get_base_path(id)
    classification_path = os.path.join(
        base_path, "classification")
    create_dir(classification_path)
    classification = Classification(
        get_image_source_path(id), classification_path)
    classification.run()
    return JSONResponse(content={"file_id": id}, status_code=200)


@app.post('/dosegmentation/')
async def do_segmentation(id: str = Form(...)):
    base_path = get_base_path(id)
    segmented_path = os.path.join(
        base_path, "segmented")
    create_dir(segmented_path)
    segmentation = Segmentation(get_image_source_path(id), segmented_path)
    segmentation.run()
    return JSONResponse(content={"file_id": id}, status_code=200)


@app.get("/viewclassification/")
async def view_classifcation(directory: str):
    base_path = os.path.join(DESTINATION, directory)
    classification_path = os.path.join(
        base_path, "classification")
    data = json.load(
        open(os.path.join(classification_path, 'classification.json')))
    return JSONResponse(content=data)


@app.get("/viewsegmentation/", responses={200: {"content": {"image/png": {}}}}, response_class=Response)
async def view_segmentation(directory: str, img_num: int):
    base_path = os.path.join(DESTINATION, directory)
    segmented_path = os.path.join(
        base_path, "segmented")
    data = cv2.imread(os.path.join(
        segmented_path, '{:>05d}.png'.format(img_num)))
    res, im_png = cv2.imencode(".png", data)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):

    # Hashing
    unique_path = await sha1_hashing(file)
    base_path = os.path.join(DESTINATION, unique_path)
    if is_dir_exist(base_path):
        return {
            "file": "File Udah Ada",
            "id": unique_path
        }

    # Extract
    file_path = await save_to(file, base_path, file.filename)
    dicom_path = os.path.join(
        base_path, "dicom")

    create_dir(dicom_path)
    shutil.unpack_archive(file_path, dicom_path)
    move_dicom_to_root(dicom_path, base_path)

    # Result
    result = {
        'directory': base_path,
    }

    result = json.dumps(result)

    return JSONResponse(content=result)


@app.post("/remove/")
def remove_data(id: str = Form(...)):
    path = get_base_path(id)
    if os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        return JSONResponse(content={"status": "failed"}, status_code=404)

    return JSONResponse(content={"status": "success"})
