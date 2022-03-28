## Install DCNv2
Ref: [Installation guide](https://github.com/xingyizhou/CenterNet/issues/118)

In case of any errors, first remove DCNv2, then clone and install it again.
```sh
cd src/lib/models/networks/
rm -rf DCNv2
git clone https://github.com/CharlesShang/DCNv2
cd DCNv2
./make.sh
```

## Run demo program
```sh
cd src
python demo.py ctdet --demo ../images --load_model ../models/ctdet_coco_dla_2x.pth
```