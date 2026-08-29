import warnings, os, argparse
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # 代表用cpu训练 不推荐！没意义！ 而且有些模块不能在cpu上跑
# os.environ["CUDA_VISIBLE_DEVICES"]="0"     # 代表用第一张卡进行训练  0：第一张卡 1：第二张卡
# 多卡训练参考<使用教程.md>下方常见错误和解决方案
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

# 深度学习炼丹必备必看必须知道的小技巧！https://www.bilibili.com/video/BV1q3SZYsExc/
# 整合多个创新点的B站视频链接:
# 1. [YOLOV8-不会把多个改进整合到一个yaml配置文件里面？那来看看这个吧！从简到难手把手带你整合三个yaml](https://www.bilibili.com/video/BV15H4y1Y7a2/)
# 2. [细谈目标检测中的小目标检测头和大目标检测检测头，并教懂你怎么加微小目标、极大目标检测头！](https://www.bilibili.com/video/BV1jkDWYFEwx/)
# 3. [不会看YOLO的模型yaml配置文件？那你还怎么整合多个配置文件！](https://www.bilibili.com/video/BV1oiBRYnEEw/)
# 4. [不会把多个创新点整合到一个yaml配置文件里面？那来看看这个吧！手把手来你整合创新点！](https://www.bilibili.com/video/BV1DUBRYGE3b/)
# 更多问题解答请看使用说明.md下方<常见疑问>

# 现版本中保存模型会比之前的大一倍，原因是因为之前保存模型的格式是fp16，但是fp16可能会导致部分模型在训练中精度正常
# 但是在val.py的时候精度为0，所以统一改成fp32，原则上没影响，只是存储的位数变多了。

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml', help='model yaml or pt path')
    parser.add_argument('--name', type=str, default='baseline', help='experiment name')
    parser.add_argument('--device', type=str, default='', help='cuda device, e.g. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()

    model = RTDETR(args.model)
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=350,
                patience=50,
                batch=4,
                workers=4,
                pretrained=False,
                resume='',
                device=args.device,
                project='runs/train',
                name=args.name,
                )