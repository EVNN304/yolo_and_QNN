
from Module_save_detection_v2 import *

class Yolo_inits_batch:
    def __init__(self, q_from_neuro: mp.Queue, saved_mode=None, names_files=None, name_folder="TEST_SAVE_SERVER_", classes_naames={}):
        self.pix_x, self.pix_y = 30, 30
        self.name_window = "w"
        self.name_folder = name_folder
        self.q_from_neuro = q_from_neuro
        self.saved_mode = saved_mode
        self.udp_caster = None
        self.names_files = names_files
        self.name_classes = classes_naames

        if self.saved_mode != None:
            print("RUNNNER")
            self.saved_mode = mp.JoinableQueue(1)
            self.run_saver()




    def set_name_window(self, val):
        self.name_window += val

    def set_name_save(self, val:str):
        self.names_files = val

    def set_name_folder(self, val):
        self.name_folder = val

    def set_self_addres(self, addr):
        self.self_addres = addr

    def set_dest_addres(self, addr):
        self.dest_address = addr


    def set_pix_x(self, val):
        self.pix_x = val


    def set_pix(self, val):
        self.pix_y = val



    def run_saver(self):
        process_save_det = Save_detect(self.saved_mode, classes_names=self.name_classes)
        process_save_det.set_names_file(self.names_files)
        process_save_det.set_path_save(self.name_folder)

        process_save_det.main_start_process()
        print("STARTED", self.name_classes)


    def run_nets(self):
        neuro_data_collector = mp.Process(target=self.collect_n_cast_neuro, args=(self.q_from_neuro, self.saved_mode))
        neuro_data_collector.start()



    def collect_n_cast_neuro(self, q_dets: mp.Queue, saved_mode):


        #timer = Timer()
        start_time = time.time()
        fr_c = 0
        while True:
            #timer.start()

            if not q_dets.empty():
                frame, detections = q_dets.get()
                copy_img = copy.deepcopy(frame)
                fr_c += 1
                for det_n, det in enumerate(detections):
                    x_lft, y_lft = det.left_top()
                    x_rght, y_rght = det.right_bottom()
                    conf, obj = det.p, det.obj_class
                    cv.rectangle(frame, (x_lft, y_lft), (x_rght, y_rght), (0, 0, 255), 6)
                    cv.putText(frame, f'{self.name_classes[obj]}[{round(conf, 3)}]', (x_lft, y_lft - 10), 3, 1.3, (0, 0, 255),2)
                #saved_mode.put([fr, filtered_dets, True])
                #print("MODES", saved_mode != None)
                current_time = time.time()

                fps = 1.0 / (current_time - start_time) if fr_c > 0 else 0.0  # Мгновенный FPS
                start_time = current_time
                print(f"FPS_DET: {fps}")
                if saved_mode != None:
                    if saved_mode.empty():
                        if detections:
                            saved_mode.put([copy_img, detections, True])

                cv.imshow(self.name_window, cv.resize(frame, (700, 700)))
                cv.waitKey(1)