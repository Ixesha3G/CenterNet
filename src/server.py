import cv2
import numpy as np
import os.path, sys
import base64
sys.path.insert(0, './CenterNet/src/lib/')

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse
from opts import opts
from detectors.detector_factory import detector_factory
from utils.debugger import Debugger
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL_PATH = './CenterNet/models/ddd_3dop.pth'
TASK = 'ddd'
opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
detector = detector_factory[opt.task](opt)

@app.post("/camera/uploadcamera/")
async def creat_camera_img(request: Request):
    payload = await request.json()
    camera_b64 = payload['camera']
    nparr = np.fromstring(base64.b64decode(camera_b64.split(',')[1]), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img1 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # HxWxC
    # for i in range(2):
    #     ret = detector.run(img, debug = i)['results']
    #     detector.show_results(Debugger(dataset=opt.dataset), img, ret, debug = i)
    ret = detector.run(img)['results']
    detector.show_results(Debugger(dataset=opt.dataset), img, ret)
    ret1= detector.run(img1, debug=1)['results']
    detector.show_results(Debugger(dataset=opt.dataset), img1, ret1, debug=1)
    img_ID = 0
    if os.path.exists("./outputCenterNet/id.txt"):
    	img_ID = int(open("./outputCenterNet/id.txt", "r").read()) - 1
    response_img = "./outputCenterNet/{}add_pred.png".format(img_ID)
    return FileResponse(response_img)

@app.get("/service/getPhotosNumber/")
def get_photos_number():
    if os.path.exists("./outputCenterNet/id.txt"):
        return {"photos_number": int(open("./outputCenterNet/id.txt", "r").read())}
    else:
        return {"photos_number": 0}

@app.get("/service/getOldPhotos/{photo_id}")
def read_photos(photo_id: int):
    response_img = "./outputCenterNet/{}add_pred.png".format(photo_id - 1)
    return FileResponse(response_img)


@app.post("/service/uploadfile/")
async def create_upload_file(file: UploadFile):
    img = await file.read()
    img_buffer = np.frombuffer(img, dtype=np.uint8)
    img_numpy = cv2.imdecode(img_buffer, 1)
    img_numpy1 = cv2.imdecode(img_buffer, 1)
    # for i in range(2):
    #     ret = detector.run(img_numpy, debug=i)['results']
    #     detector.show_results(Debugger(dataset=opt.dataset), img_numpy, ret, debug=i)
    ret = detector.run(img_numpy)['results']
    detector.show_results(Debugger(dataset=opt.dataset), img_numpy, ret)
    ret1= detector.run(img_numpy1, debug=1)['results']
    detector.show_results(Debugger(dataset=opt.dataset), img_numpy1, ret1, debug=1)
    img_ID = 0
    if os.path.exists("./outputCenterNet/id.txt"):
    	img_ID = int(open("./outputCenterNet/id.txt", "r").read()) - 1
    response_img = "./outputCenterNet/{}add_pred.png".format(img_ID)
    return FileResponse(response_img)
