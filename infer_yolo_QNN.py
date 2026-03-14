import sys
import multiprocessing as mp
import cv2
import numpy as np
sys.path.insert(0, '/home/grida/PycharmProjects/yolo/ultralytics')



def set_cam_param(cap, set_w, set_h):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, set_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, set_h)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    #cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H','2','6','4'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    #print(cap.get(cv2.CAP_PROP_FPS), cap.getBackendName())

##
def process_video(q_vid:mp.Queue, q_set:mp.Queue, flag_set, path_video):


    cap = cv2.VideoCapture(path_video)
    set_w, set_h = None, None
    if flag_set:
        try:
            set_w, set_h = q_set.get()
            set_cam_param(cap, set_w, set_h)
        except Exception as e:
            print(f"Errr_set_param_cam:{e.args}")

    c = 0

    while True:
        try:
            flag, image = cap.read()
            if flag:
                c = 0
                #if q_vid.empty():
                q_vid.put(image)
        except Exception as e:
            print(f"Errr_connect_cam_or_video: {e.args}")
            cap.release()
            cv2.destroyAllWindows()
            time.sleep(3)

            cap = cv2.VideoCapture(path_video)
            if flag_set and cap.isOpened():
                set_cam_param(cap, set_w, set_h)
            c += 1
            print(f"Reconn_cam/video: {cap.isOpened()}, count try connect: {c}")
            continue

if __name__ == '__main__':

    mp.set_start_method('spawn', force=True)
    import sys
    from yolo_batch_main_mot import *
    from init_Yolo_for_sahi_batches_v2 import *
    from infer_pennyline_yolo import *

    crop_w, crop_h = 288, 288
    overlay_w, overlay_h = 0.1, 0.1
    set_w, set_h = 1920, 1200


    #path_video = f"/home/grida/PycharmProjects/yolo/ultralytics/2_n.avi"
    path_video = f"/home/grida/Документы/video/stock-footage-a-group-of-combat-helicopters-performed-a-simultaneous-demonstration-flight-at-the-air-show.webm"


    flag_use_denoise = False
    cap_flag_set = False
    flag_video = True
    pth = f"/home/usr/sahi3/" # путь к картинкам
    QUANTUM_MODEL_PATH = "/home/grida/PycharmProjects/HQNN_class/drones_model_3.pth"
    q_video = mp.Queue(maxsize=1)
    q_set = mp.Queue(maxsize=1)
    q_to_qnn = mp.Queue(maxsize=1)
    q_in, q_out =  mp.Queue(maxsize=1), mp.Queue(maxsize=1)
    q_send_names = mp.Queue(maxsize=1)
    q_quantum_out = mp.Queue(maxsize=1)
    print(pth)
                                                                        # /home/usr/Рабочий стол/weights_yolo26/drone_iter_3_m/train23/weights/best.pt
    cl = Yolo_batches(q_in, q_out, q_send_names, q_to_qnn)   # /home/usr/PycharmProjects/yolo_proj/ultralytics/runs/detect/train20/weights/best.pt    /home/usr/Рабочий стол/weights_yolo26/drone_iter_2/train22/weights/best.pt
    cl.set_path_model("/home/grida/PycharmProjects/yolo/ultralytics/best_yolo11x_288x288_batch_64.pt")   #### /home/usr/PycharmProjects/yolo_proj/ultralytics/final_weights_train/ground_to_air/best_yolo11x_288x288_batch_64.pt
    cl.set_size_inp_layers(288)
    cl.set_conf_model(0.6)
    print("SIZE_LAYERS", cl.get_size_inp_layers())
    cl.process_start()
    names_classes = q_send_names.get()

    print("Classes_load_model:", names_classes)


    neuro_start_proc = Yolo_inits_batch(q_out, names_files="", saved_mode=None, name_folder="/home/usr/Изображения/saver_detect_7777777_v6", classes_naames=names_classes)
    cl.set_nms_type("classic")
    neuro_start_proc.run_nets()

    print("[MAIN] Initializing Quantum processor...")
    quantum_proc = Quantum_batches(
        q_from_yolo=q_to_qnn,
        q_out=q_quantum_out,
        path_model="/home/grida/PycharmProjects/HQNN_class/drones_model_3.pth",
        class_map={0: "drone", 1: "bird", 2: "plane", 3: "background"},
        num_classes=4,
        verbose=True
    )
    quantum_proc.set_size_inp_layers(224)
    quantum_proc.set_conf_model(0.6)
    quantum_proc.process_start(daemon=True)


    print("[MAIN] Initializing Visualizer...")
    visualizer = QuantumVisualizer(
        q_results=q_quantum_out,
        window_name="Quantum-Enhanced Detection",
        window_size=(1280, 720)
    )
    visualizer_proc = visualizer.process_start(daemon=True)




    kernel = np.ones((4, 4), dtype=np.float32) / 17
    if flag_video:

        proc_vid = mp.Process(target=process_video, args=(q_video, q_set, cap_flag_set, path_video), daemon=False)
        proc_vid.start()
        if cap_flag_set:
            q_set.put([set_w, set_h])
        time.sleep(4)
        if not q_video.empty():
            img = q_video.get()
            h_p, w_p, _ = img.shape
        else:
            flg, img = False, None
            exit(0)


        fps_timer = Timer()
        fps_timer.start()

        lst_cord = calc_scan_areas([0, 0, w_p, h_p], window_w_h=(crop_w, crop_h), overlay=(overlay_w, overlay_h))
        prev_frames_for_motion_est = []  # Для оценки движения (опционально)
        max_prev_frames = 1
        while True:
            if not q_video.empty():

                image = q_video.get()
                ##image[900:1080, 20:350, :] = 0
                list_image, list_cropp_cord = [], []
                for i, k in enumerate(lst_cord):
                    fragment_processed = image[k[1]:k[3], k[0]:k[2]]

                    list_image.append(fragment_processed)
                    list_cropp_cord.append([k[0], k[1]])
                    #q_in.put((image[k[1]:k[3], k[0]:k[2]], [k[0], k[1], k[2], k[3]], [frame_count, len([[k[0], k[1], k[2], k[3]]]), 1, cnn]))

                list_image.append(image)
                list_cropp_cord.append([0, 0])
                if q_in.empty():
                    q_in.put([list_image, list_cropp_cord])


                elapsed = fps_timer.stop()
                fps_timer.start()
                print(f'capture fps: {1 / elapsed if elapsed != 0 else 1}')
                cv.imshow('original', cv.resize(image, (600, 600)))

                cv.waitKey(100)


    else:
        lst = os.listdir(pth)

        img = cv2.imread(pth+lst[0])
        h_p, w_p, _ = img.shape

        fps_timer = Timer()
        fps_timer.start()

        lst_cord = calc_scan_areas([0, 0, w_p, h_p], window_w_h=(crop_w, crop_h), overlay=(overlay_w, overlay_h))
        for i, k in enumerate(lst):


            image = cv2.imread(pth+k)
            print(image.shape)
            list_image, list_cropp_cord = [], []
            for i, k in enumerate(lst_cord):
                list_image.append(image[k[1]:k[3], k[0]:k[2]])
                list_cropp_cord.append([k[0], k[1]])
            list_image.append(image)
            list_cropp_cord.append([0, 0])
            if q_in.empty():
                q_in.put([list_image, list_cropp_cord, image])



            elapsed = fps_timer.stop()
            fps_timer.start()
            print(f'capture fps: {1 / elapsed if elapsed != 0 else 1}')
            cv.imshow('original', cv.resize(image, (600, 600)))

            cv.waitKey(0)