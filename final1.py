import cv2
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops
from trackers.tracker import Tracker
import copy
import argparse
import numpy as np
import time
from ultralytics import YOLO
from utils.etc import *

def make_parser():
    parser = argparse.ArgumentParser("Tracker")

    # For detectors
    parser.add_argument("--model", type=str, default="yolov8n.engine", help="模型权重")
    parser.add_argument("--conf", type=float, default=0.6, help="置信度")
    parser.add_argument("--iou", type=float, default=0.85, help="交并比")
    parser.add_argument("--classes", type=int, default=0, help="仅检测人")

    # For trackers
    parser.add_argument("--min_len", type=int, default=1, help="定义新轨迹的成熟时间，出现3秒才进行跟踪(state:new->tracked)")  #
    parser.add_argument("--max_time_lost", type=float, default=60, help="tracktrack默认为帧数*2")
    parser.add_argument("--penalty_p", type=float, default=0.20, help="检测置信度为low的代价矩阵的惩罚项")
    parser.add_argument("--penalty_q", type=float, default=0.40, help="检测置信度为high但在nms中被删的代价矩阵的惩罚项")
    parser.add_argument("--reduce_step", type=float, default=0.05, help="匹配阶段，每次匹配的阈值减少的大小，逐次缩减，越来越严格。")
    parser.add_argument("--tai_thr", type=float, default=0.55, help="track_aware_nms时用到，用于判断两个目标是否重叠过高")
    parser.add_argument("--gmc_method", type=str, default='sparseOptFlow', help="相机补偿Options include 'orb', 'sift', 'ecc', 'sparseOptFlow', 'none'")

    # For algorithm
    parser.add_argument("--det_thr", type=float, default=0.60, help="从det0.80区分高分检测 和 低分检测 的标准值")
    parser.add_argument("--init_thr", type=float, default=0.50, help="轨迹初始化时使用，检测轨迹置信度，低于此值会被直接过滤")
    parser.add_argument("--match_thr", type=float, default=0.70,help="轨迹和检测框关联必须达到的iou匹配值，会逐步缩减，提高要求，跟reduce_step配合使用")

    # For ReID
    parser.add_argument("--with_ReID", type=bool, default=False,help="是否使用ReID")
    parser.add_argument("--ReID_config", type=bool, default=False,help="ReID模型配置")
    parser.add_argument("--ReID_weight", type=bool, default=False,help="ReID权重")


    # For vedio test
    parser.add_argument("--video", type=str, required=False,default=r"G:\需求\dataset\MOT17\train\video\mot17视频\MOT17-09-64.mp4", help="输入视频路径")  # 新增视频路径参数

    return parser


def main(args):

    overrides = {
        "model": args.model,
        "conf": args.conf,
        "iou": args.iou,
        "classes": args.classes,
    }

    model = YOLO(args.model)
    # 手动初始化ultralytics里的YOLO的predictor，因为U库提供的接口无法满足需求。
    if not model.predictor:
        model.predictor = DetectionPredictor(overrides=overrides)
        model.predictor.setup_model(args.model)

    # 初始化跟踪器
    tracker = Tracker(args)

    # 初始化摄像头
    cap = cv2.VideoCapture(0) # 0为设备号

    # 修改为读取视频文件
    # cap = cv2.VideoCapture(args.video)
    # if not cap.isOpened():
    #     raise ValueError(f"无法打开视频文件: {args.video}")

    # For each frame
    results = [] # 用于存放跟踪结果，帧id、trackID、x1x2y1y2、conf、-1、-1、-1
    frame_id = 0

    # FPS计算相关变量
    prev_time = 0


    while True:
        ret, frame = cap.read()
        if not ret:
            break
        original_frame = frame.copy()

        # 计算FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        with model.predictor._lock:  # for thread-safe inference
            # Setup source every time predict is called
            model.predictor.setup_source(frame)

            # Warmup model
            if not model.predictor.done_warmup:
                model.predictor.model.warmup(
                    imgsz=(1 if model.predictor.model.pt or model.predictor.model.triton else model.predictor.dataset.bs,
                           model.predictor.model.ch, *model.predictor.imgsz)
                )
                model.predictor.done_warmup = True

            model.predictor.seen, model.predictor.windows, model.predictor.batch = 0, [], None

            model.predictor.run_callbacks("on_predict_start") # 目前不做任何事情
            for model.predictor.batch in model.predictor.dataset:
                model.predictor.run_callbacks("on_predict_batch_start") # 目前不做任何事情
                paths, im0s, s = model.predictor.batch

                # Preprocess
                im = model.predictor.preprocess(im0s)

                # Inference
                preds,_ = model.predictor.inference(im)
                preds_clone = copy.deepcopy(preds)

                # Postprocess
                model.predictor.results = model.predictor.postprocess(preds, im, im0s)
                model.predictor.args.iou = 0.95 # 用于筛选出被nms过滤的候选框
                model.predictor.results095 = model.predictor.postprocess(preds_clone, im, im0s)  # 这将会在predictor上增加一个results095属性

        # 跟踪
        det080 = model.predictor.results[0].boxes.cpu().numpy()
        if det080 is not None and len(det080) > 0:
            det095 = model.predictor.results095[0].boxes.cpu().numpy()
            # 处理结果
            det080 = concatenate_det_arrays(det080)
            det095 = concatenate_det_arrays(det095)
            # 获取特征
            feat080 = model.predictor.results[0].feats.cpu().numpy()
            feat095 = model.predictor.results095[0].feats.cpu().numpy()
            # 拼接结果和特征
            det080 = np.concatenate([det080,feat080], axis=1)
            det095 = np.concatenate([det095,feat095], axis=1)

            tracks = tracker.update(det080, det095,frame)
        else:
            tracks = tracker.update_without_detections(frame)


        # 可视化轨迹
        x1y1whs, track_ids, scores = [], [], []
        print("匹配轨迹数量：", len(tracks))
        for t in tracks:
            x1y1whs.append(t.x1y1wh)
            track_ids.append(t.track_id)
            scores.append(t.score)

        frame_id +=1
        results.append([frame_id, track_ids, x1y1whs, scores])
        # 可视化跟踪结果
        visualized_frame = visualize_tracking_results(original_frame.copy(), [track_ids, x1y1whs, scores])

        # 在画面上显示FPS
        cv2.putText(visualized_frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 显示结果
        cv2.imshow("Tracking Results", visualized_frame)

        if cv2.waitKey(1) == ord("q"):  # 按 Q 退出
            break

    write_results("./outputs/res.txt",results)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)

# 同个目标多个框

