import cv2
import numpy as np
import os.path, sys
CENTERNET_PATH = './lib/'
sys.path.insert(0, CENTERNET_PATH)

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from detectors.detector_factory import detector_factory
from opts import opts
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

MODEL_PATH = '../models/ddd_3dop.pth'
TASK = 'ddd'
opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
detector = detector_factory[opt.task](opt)

@app.post("/service/uploadfile/")
async def create_upload_file(file: UploadFile):
    img = await file.read()
    img_buffer = np.frombuffer(img, dtype=np.uint8)
    img_numpy = cv2.imdecode(img_buffer, 1)
    ret = detector.run(img_numpy)['results']
    detector.show_results(Debugger(dataset=opt.dataset), img_numpy, ret)
    if os.path.exists("../../../outputCenterNet/id.txt"):
    	img_ID = int(open("../../../outputCenterNet/id.txt", "r").read()) - 1
    else:
    	img_ID = 0
    response_img = "../../../outputCenterNet/{}add_pred.png".format(img_ID)
    return FileResponse(response_img)
