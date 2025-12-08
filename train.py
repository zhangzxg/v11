import warnings, os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
# os.environ["CUDA_VISIBLE_DEVICES"]="0"     # 代表用第一张卡进行训练  0：第一张卡 1：第二张卡
# 多卡训练参考<YOLOV11配置文件.md>下方常见错误和解决方案
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('v11-small-full.yaml') # YOLO11 完整模型（所有改进点）
    # model.load('yolo11n.pt') # loading pretrain weights
    model.train(
	data='/root/autodl-tmp/v11/v11-data.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=16,  # 减小batch size以减少内存占用（从16改为8）
                close_mosaic=0, # 最后多少个epoch关闭mosaic数据增强，设置0代表全程开启mosaic训练
                workers=4, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                # device='0,1', # 指定显卡和多卡训练参考<YOLOV11配置文件.md>下方常见错误和解决方案
                optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt,不懂就在百度云.txt找断点续训的视频
                amp=True, # close amp | loss出现nan可以关闭amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )