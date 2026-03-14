import socket
import v_packet_constructor as p_cons
import cv2 as cv
from toolset import Mp_dev_interface, Command
import multiprocessing as mp
import copy
import threading as thr
import numpy as np
import time
from record_tools import Recorder
# from pelco import Mp_dev_interface


TCP = 0
UDP = 1

class CMD_list:
    set_mode = 0
    get_mode = 1

    got_tracks = 4 #[meta, simple_tracks]

    get_frame = 51  #[frame_meta]
    get_det = 52    #[frame_meta]
    get_frame_det = 53  #[frame_meta]

    get_area_det = 54 #[frame_meta, areas]
    get_area_frame_det = 55  # [frame_meta, areas]

    get_t_recognitions = 56 # [stamp, [[id0,az0,el0],[id1,az1,el1]....]


    got_r_frame = 41    #[frame_meta,frame]
    got_det = 42        #[frame_meta, detections]
    got_frame_det = 43  #[frame_meta, frame, detections]
    got_recgs_imgs = 45  # [frame_meta,dets,imgs]

    udp_set_server_mode = 60 # Пакет для изменения режима работы сервера
    udp_set_cam_settings = 61   #Пакет для изменения настроек камеры
    udp_set_add_settings = 62   #Пакет для измениня настроек обработчиков

    serv_goto_cmd = 71 #Пакет для поворота в указанном направлении

    udp_locator_tracks = 101
    udp_locator_tracks_ref = 110 #Обновленная версия протокола получения траекторий
    udp_locator_current_az = 102
    udp_camera_direction = 103
    udp_locator_stat = 104 #[Az, T, Opu_Stat, Signal_proc_stat, Reserv_1, Reserv_2]
    udp_locator_control = 105 #[on/off, reserv_1, reserv_2, reserv_3]

    udp_simple_targeting = 123 #[message_type]
    udp_simple_recognition = 124
    udp_ext_targeting = 125

    udp_contours = 131 #[frame_meta, contours]

    send_simple_image = 111 #[frame_meta,image] message type 1 (сжатое изображение без отметок)



class Simple_udp_dialog:
    def __init__(self, address, dest_address = ('localhost', 5055)):
        self.autodetect_dst = False
        self.cmd_list = CMD_list
        self.sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.ip = address[0]
        self.port = address[1]
        self.interface = Mp_dev_interface(2,allow_loose=True)
        self.dest_addr = dest_address
        self.constructor = p_cons.V_constructor()
        # self.sock.connect(self.dest_addr)

        self.bufferSize = 4096
        self.log_processor_rx = None
        # Bind to address and ip

    def encode_targeting(self,obj_id,stamp,az,el,tail_l):
        buf = bytearray(18)
        buf[0:1] = int(self.cmd_list.udp_simple_targeting).to_bytes(2, 'little', signed=False)
        buf[2:5] = int(obj_id).to_bytes(4, 'little', signed=False)
        buf[6:9] = int(stamp).to_bytes(4, 'little', signed=False)
        buf[10:12] = int(az * 100).to_bytes(3, 'little', signed=True)
        buf[13:15] = int(el * 100).to_bytes(3, 'little', signed=True)
        buf[16:17] = int(tail_l).to_bytes(2, 'little', signed=False)

        return buf

    def encode_recognition(self,obj_id,stamp,az,el,obj_class):
        buf = bytearray(18)
        buf[0:1] = int(self.cmd_list.udp_simple_recognition).to_bytes(2, 'little', signed=False)
        buf[2:5] = int(obj_id).to_bytes(4, 'little', signed=False)
        buf[6:9] = int(stamp).to_bytes(4, 'little', signed=False)
        buf[10:12] = int(az * 100).to_bytes(3, 'little', signed=True)
        buf[13:15] = int(el * 100).to_bytes(3, 'little', signed=True)
        buf[16:17] = int(obj_class).to_bytes(2, 'little', signed=False)

        return buf

    def encode_ext_targeting(self,obj_id,stamp,az,el,obj_class,r_plane,r_3d,h):
        buf = bytearray(24)
        buf[0:1] = int(self.cmd_list.udp_ext_targeting).to_bytes(2, 'little', signed=False)
        buf[2:5] = int(obj_id).to_bytes(4, 'little', signed=False)
        buf[6:9] = int(stamp).to_bytes(4, 'little', signed=False)
        buf[10:12] = int(az * 100).to_bytes(3, 'little', signed=True)
        buf[13:15] = int(el * 100).to_bytes(3, 'little', signed=True)
        buf[16:17] = int(obj_class).to_bytes(2, 'little', signed=False)
        buf[18:19] = int(r_plane).to_bytes(2, 'little', signed=False)
        buf[20:21] = int(r_3d).to_bytes(2, 'little', signed=False)
        buf[22:23] = int(h).to_bytes(2, 'little', signed=True)
        return buf

    def decode_message(self,buffer,offset = 0):
        try:
            message_id = int.from_bytes(buffer[offset:offset+1], 'little', signed=False)
            params = []
            offset += 2
            if message_id == self.cmd_list.udp_simple_targeting:

                obj_id = int.from_bytes(buffer[offset:offset+3], 'little', signed=False)
                stamp = int.from_bytes(buffer[offset+4:offset+7], 'little', signed=False)
                az = int.from_bytes(buffer[offset+8:offset+10], 'little', signed=False) / 100
                el = int.from_bytes(buffer[offset+11:offset+13], 'little', signed=True) / 100
                tail_l = int.from_bytes(buffer[offset+14:offset+15], 'little', signed=False)
                params = [obj_id,stamp,az,el,tail_l]
            elif message_id == self.cmd_list.udp_simple_recognition:
                obj_id = int.from_bytes(buffer[offset:offset + 4], 'little', signed=False)
                stamp = int.from_bytes(buffer[offset + 4:offset + 8], 'little', signed=False)
                az = int.from_bytes(buffer[offset + 8:offset + 11], 'little', signed=False) / 100
                el = int.from_bytes(buffer[offset + 11:offset + 14], 'little', signed=True) / 100
                obj_class = int.from_bytes(buffer[offset + 14:offset + 16], 'little', signed=False)
                params = [obj_id, stamp, az, el, obj_class]
                print('!__________Получено сообщение о распознавании')
            elif message_id == self.cmd_list.udp_ext_targeting:
                obj_id = int.from_bytes(buffer[offset:offset + 4], 'little', signed=False)
                stamp = int.from_bytes(buffer[offset + 4:offset + 8], 'little', signed=False)
                az = int.from_bytes(buffer[offset + 8:offset + 11], 'little', signed=False) / 100
                el = int.from_bytes(buffer[offset + 11:offset + 14], 'little', signed=True) / 100
                obj_class = int.from_bytes(buffer[offset + 14:offset + 16], 'little', signed=False)
                r_plane = int.from_bytes(buffer[offset + 16:offset + 18], 'little', signed=False)
                r_3d = int.from_bytes(buffer[offset + 18:offset + 20], 'little', signed=False)
                h = int.from_bytes(buffer[offset + 20:offset + 22], 'little', signed=True)
                params = [obj_id,stamp,az,el,obj_class,r_plane,r_3d,h]
            else:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!Принято неопознанное сообщение!!!!!!!!',message_id)

            return message_id,params
        except Exception as e:
            print('UDP message decode exception ',e.args)
            return -1,[]






    def run_dialog(self):
        run = True
        bufferSize = self.bufferSize
        self.sock.bind((self.ip, self.port))
        self.sock.settimeout(0.02)
        # self.sock.connect(self.dest_addr)
        th = thr.Thread(target=self.write_proc, args = ())
        th.daemon = True
        th.start()
        while run:
            try:

                bytesAddressPair = self.sock.recvfrom(bufferSize)
                message = bytesAddressPair[0]
                print(f'UDP ALIVE: {self.ip,self.port}')
                if message:
                    if self.log_processor_rx:
                        self.log_processor_rx.add_record_no_session(time.time(),message,'udp')

                    # print('!')
                    address = bytesAddressPair[1]
                    # if self.autodetect_dst:
                    #     self.dest_addr = address
                    print(f'received {self.ip,self.port} from {bytesAddressPair[1]}')
                    # self.dest_addr = address
                    mes_id, params = self.decode_message(message,0)
                    # print(f'simple header: {mes_id}')
                    # obj_id = int.from_bytes(message[0:3], 'little', signed=False)
                    # stamp = int.from_bytes(message[4:7], 'little', signed=False)
                    # az = int.from_bytes(message[8:10], 'little', signed=False) / 100
                    # el = int.from_bytes(message[11:13], 'little', signed=True) / 100
                    # tail_l = int.from_bytes(message[14:15], 'little', signed=False)
                    # self.interface.push_rep_from_dev(Command('got_message', [obj_id,stamp,az,el,tail_l,message]))
                    if mes_id == self.cmd_list.udp_simple_targeting:
                        self.interface.push_rep_from_dev(Command('got_targeting', params))
                    elif mes_id == self.cmd_list.udp_simple_recognition:
                        self.interface.push_rep_from_dev(Command('got_recognition',params))
                    elif mes_id == self.cmd_list.udp_ext_targeting:
                        self.interface.push_rep_from_dev(Command('got_ext_targeting', params))
                    else:
                        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!UnKnown_ CMD!!!!!!!!!!!!!!!', len(message))
                        header, offset = self.constructor.parse_header(message, 0)
                        # print(header)
                        # print('full header:')
                        # header.print()
                        if header.message_type == 131:
                            meta_r, dets_r, offset = self.constructor.parse_message_131(message, offset)
                            # print(r'//--\\')
                            # meta_r.print()
                            # print(dets_r)
                            self.interface.push_rep_from_dev(Command('got_raw', [meta_r,dets_r]))
                        elif header.message_type == 116:
                            params = (message.decode()).split(',')
                            self.interface.push_rep_from_dev(Command('unknown',[message]))
                            print(f'got 116: {message}')
                            header, offset = self.constructor.parse_header(message)
                            # header.print()
                        elif header.message_type == CMD_list.udp_locator_tracks:
                            # self.interface.push_rep_from_dev(Command('locator', [message]))
                            # print(f'got 101')
                            locator_tracks_list,offset = self.constructor.parse_message_101(message,offset)
                            self.interface.push_rep_from_dev(Command(CMD_list.udp_locator_tracks,locator_tracks_list))
                            # header, offset = self.constructor.parse_header(message)
                            # header.print()
                        elif header.message_type == CMD_list.udp_locator_tracks_ref:
                            locator_tracks_list, offset = self.constructor.parse_massage_110(message, offset)
                            self.interface.push_rep_from_dev(Command(CMD_list.udp_locator_tracks, locator_tracks_list))

                        elif header.message_type == CMD_list.udp_locator_current_az:
                            current_az, opu_t, offset = self.constructor.parse_message_102(message,offset)
                            self.interface.push_rep_from_dev(Command(CMD_list.udp_locator_current_az,[current_az,opu_t]))
                        elif header.message_type == CMD_list.udp_locator_stat:
                            locator_stats, offset = self.constructor.parse_message_104(message,offset)
                            self.interface.push_rep_from_dev(Command(CMD_list.udp_locator_stat,locator_stats))
                        elif header.message_type == 1:
                            meta, image, offset = self.constructor.parse_message_1(message,offset)
                            self.interface.push_rep_from_dev(Command(CMD_list.got_r_frame,[meta,image]))
                        elif header.message_type == 2:
                            meta, dets, offset = self.constructor.parse_message_2(message, offset)
                            self.interface.push_rep_from_dev(Command(CMD_list.got_det, [meta, dets]))
                        elif header.message_type == 4:
                            meta, tracks, offset = self.constructor.parse_message_4(message, offset)
                            self.interface.push_rep_from_dev(Command(CMD_list.got_tracks, [meta, tracks]))
                        elif header.message_type == 56:
                            stamp,targetings,offset = self.constructor.parse_t_recognition_req(message,offset)
                            self.interface.push_rep_from_dev(Command(CMD_list.get_t_recognitions,[stamp,targetings]))
                        elif header.message_type == 5:
                            meta, dets_dec, images, offset = self.constructor.parse_message_5(message, offset)
                            self.interface.push_rep_from_dev(Command(CMD_list.got_recgs_imgs,[meta,dets_dec,images]))
                        elif header.message_type == 60:
                            stamp, mode, reserved, offset = self.constructor.parse_message_60(message, offset)
                            self.interface.push_rep_from_dev(
                                Command(CMD_list.udp_set_server_mode, [stamp, mode, reserved]))
                        elif header.message_type == 61:
                            flags,vals,offset = self.constructor.parse_message_61(message,offset)
                            self.interface.push_rep_from_dev(Command(CMD_list.udp_set_cam_settings,[flags,vals]))
                        elif header.message_type == 62:
                            vals,offset = self.constructor.parse_message_62(message, offset)
                            self.interface.push_rep_from_dev(Command(CMD_list.udp_set_add_settings,[vals]))
                        elif header.message_type == 71:
                            stamp, direction, reserved, offset = self.constructor.parse_message_71(message, offset)
                            self.interface.push_rep_from_dev(
                                Command(CMD_list.serv_goto_cmd, [stamp, direction, reserved]))
                            print('GoGOGOGOGOGo ',direction)
                        else:
                            self.interface.push_rep_from_dev(Command('__unknown__', [message]))
                            print(f'unable to decode message, {header.message_type}')

            except Exception as e:

                if 'timed out' in e.args:
                    pass
                else:
                    print('UDP read exception')
                    print(e.args)

    def write_proc(self):
        run = True
        while run:
            try:
                got, cmd = self.interface.get_cmd_to_dev()
                if got:
                    # cmd.print()
                    if cmd.name == 'send_targeting':
                        self.sock.sendto(b'ok', self.dest_addr)
                        # print('send targeting')
                    elif cmd.name == 'bypass':
                        self.sock.sendto(cmd.value[0], self.dest_addr)
                        # print(f'send to {self.dest_addr}, {cmd.value[0]}')
                    elif cmd.name == 'send_contours':
                        # cmd.value[0].print()
                        # print(cmd.value[1])
                        self.sock.sendto(self.constructor.build_message_131(cmd.value[1],cmd.value[0],self.constructor.DET_STYLE_BBOX),self.dest_addr)
                    elif cmd.name == self.cmd_list.udp_camera_direction:
                        self.sock.sendto(self.constructor.build_message_103(cmd.value[0],cmd.value[1]),self.dest_addr)
                    elif cmd.name == self.cmd_list.udp_locator_control:
                        self.sock.sendto(self.constructor.build_message_105(cmd.value),self.dest_addr)
                    elif cmd.name == self.cmd_list.got_det:
                        self.sock.sendto(self.constructor.build_message_2(cmd.value[1],cmd.value[0]),self.dest_addr)
                    elif cmd.name == self.cmd_list.get_t_recognitions:
                        print(f'SOCK sending targ from {self.ip},{self.port} to {self.dest_addr}')

                        self.sock.sendto(self.constructor.build_t_recognition_req(cmd.value[0], cmd.value[1]), self.dest_addr)
                    elif cmd.name == self.cmd_list.got_tracks:
                        self.sock.sendto(self.constructor.build_message_4(cmd.value[1],cmd.value[0]),self.dest_addr)
                    elif cmd.name == self.cmd_list.got_recgs_imgs:
                        self.sock.sendto(self.constructor.build_message_5(cmd.value[1],cmd.value[2],cmd.value[0]),self.dest_addr)
                    elif cmd.name == self.cmd_list.udp_set_cam_settings:
                        if len(cmd.value)>2:
                            dest_addr = cmd.value[2]
                        else:
                            dest_addr = self.dest_addr
                        self.sock.sendto(self.constructor.build_message_61(cmd.value[0],cmd.value[1]),dest_addr)
                        # pass
                        # print(f'999 dest: {dest_addr}')
                    elif cmd.name == self.cmd_list.udp_set_add_settings:
                        if len(cmd.value)>1:
                            dest_addr = cmd.value[1]
                        else:
                            dest_addr = self.dest_addr
                        self.sock.sendto(self.constructor.build_message_62(cmd.value[0]),dest_addr)
                        # pass
                    elif cmd.name == CMD_list.serv_goto_cmd:
                        stamp = cmd.value[0]
                        direction = cmd.value[1]
                        reserved= cmd.value[2]
                        if len(cmd.value)>3:
                            dest_addr = cmd.value[3]
                        else:
                            dest_addr = self.dest_addr
                        # buffer = self.constructor.build_message_71(stamp, direction[0], direction[1], reserved)
                        print('SO go on and FUCK')
                        self.sock.sendto(self.constructor.build_message_71(stamp, direction[0], direction[1], reserved),dest_addr)
                        # print(f'888 dest: {dest_addr}')
                    # elif cmd.name == self.cmd_list.send_simple_image:
                    #     # self.sock.sendto(self.constructor.build_message_1(cmd.value[1], cmd.value[0]),self.dest_addr)
                    #     self.sock.sendall(self.constructor.build_message_1(cmd.value[1],cmd.value[0]))


            except Exception as e:
                print(f'UDP write exception: {self.ip,self.port}')

                print(e.args)
    def start(self):
        pr = mp.Process(target=self.run_dialog, args = ())
        pr.daemon = True
        pr.start()
        return pr

class V_net_entity:

    def __init__(self):
        self.protocol = TCP
        self.self_ip = '127.0.0.1'
        self.self_port = 5005
        self.dest_ip = '127.0.0.1'
        self.dest_port = 5005
        self.socket = None
        self.constructor = p_cons.V_constructor()
        self.constructor.compression_q = 60
        self.q_interface = None
        self.conn = None

    def decode_message_to_interface(self, message_type,buffer,offset):
        if message_type == 51:
            meta, offset = self.constructor.parse_meta_req(buffer,offset)
            self.q_interface.push_rep_from_dev(Command(CMD_list.get_frame,[meta]))
        elif message_type == 52:
            print('"""""message 52""""""')
            meta, offset = self.constructor.parse_meta_req(buffer,offset)
            self.q_interface.push_rep_from_dev(Command(CMD_list.get_det,[meta]))
        elif message_type == 53:
            meta, offset = self.constructor.parse_meta_req(buffer,offset)
            self.q_interface.push_rep_from_dev(Command(CMD_list.get_frame_det,[meta]))
        elif message_type == 54:
            meta, areas, offset = self.constructor.parse_roi_req(buffer,offset)
            self.q_interface.push_rep_from_dev(Command(CMD_list.get_area_det,[meta,areas]))
        elif message_type == 55:
            meta, areas, offset = self.constructor.parse_roi_req(buffer,offset)
            self.q_interface.push_rep_from_dev(Command(CMD_list.get_area_frame_det,[meta,areas]))
        elif message_type == 0:
            pass
        elif message_type == 1:
            meta, image,  offset = self.constructor.parse_message_1(buffer, offset)
            self.q_interface.push_rep_from_dev(Command(CMD_list.got_r_frame, [meta, image]))
            # pass
        elif message_type == 2:
            meta, dets,offset = self.constructor.parse_message_2(buffer,offset)
            self.q_interface.push_rep_from_dev(Command(CMD_list.got_det,[meta,dets]))
        elif message_type == 3:
            meta, image, dets, offset = self.constructor.parse_message_3(buffer,offset)
            self.q_interface.push_rep_from_dev(Command(CMD_list.got_frame_det,[meta,image,dets]))
        elif message_type == 5:
            meta, dets_dec, images, offset = self.constructor.parse_message_5(buffer, offset)
            self.interface.push_rep_from_dev(Command(CMD_list.got_recgs_imgs, [meta, dets_dec, images]))
        elif message_type == 71:
            stamp, direction, reserved, offset = self.constructor.parse_message_71(buffer, offset)
            self.interface.push_rep_from_dev(Command(CMD_list.serv_goto_cmd,[stamp,direction,reserved]))

    def encode_cmd_to_buffer(self,cmd:Command):
        buffer = bytearray(0)
        if cmd.name == CMD_list.get_frame:
            meta = cmd.value[0]
            buffer = self.constructor.build_meta_req(meta,51)
        elif cmd.name == CMD_list.get_det:
            meta = cmd.value[0]
            buffer = self.constructor.build_meta_req(meta,52)
        elif cmd.name == CMD_list.get_frame_det:
            meta = cmd.value[0]
            buffer = self.constructor.build_meta_req(meta,53)
        elif cmd.name == CMD_list.get_area_det:
            meta = cmd.value[0]
            areas = cmd.value[1]
            buffer = self.constructor.build_roi_req(meta,areas,54)
        elif cmd.name == CMD_list.get_area_frame_det:
            meta = cmd.value[0]
            areas = cmd.value[1]
            buffer = self.constructor.build_roi_req(meta,areas,55)
        elif cmd.name == CMD_list.got_r_frame:
            # meta = cmd.value[0]
            buffer = self.constructor.build_message_1(cmd.value[1],cmd.value[0])
        elif cmd.name == CMD_list.got_det:
            # meta = cmd.value[0]
            buffer = self.constructor.build_message_2(cmd.value[1],cmd.value[0])
        elif cmd.name == CMD_list.got_frame_det:
            # meta = cmd.value[0]
            buffer = self.constructor.build_message_3(cmd.value[1],cmd.value[2],cmd.value[0])
        elif cmd.name == CMD_list.got_recgs_imgs:
            buffer = self.constructor.build_message_5(cmd.value[1], cmd.value[2], cmd.value[0])
        elif cmd.name == CMD_list.serv_goto_cmd:
            stamp, direction, reserved = cmd.value
            buffer = self.constructor.build_message_71(stamp,direction[0],direction[1],reserved)
            print('SO go on and FUCK')
        return buffer



    def set_protocol(self,protocol):
        self.ptotocol = TCP

    def set_self_ipport(self,ip,port):
        self.self_ip = ip
        self.self_port = port

    def set_dest_ipport(self,ip,port):
        self.dest_ip = ip
        self.dest_port = port

    def set_q_interface(self, q_interface):
        self.q_interface = q_interface

    def init_socket(self):
        pass

    def work(self):
        pass
    def run_separate_proc(self):
        proc = mp.Process(target=self.work)
        proc.start()
        proc.join()
class V_net_client(V_net_entity):

    def __init__(self):
        super().__init__()
        self.cmd_list = CMD_list()
        self.n_to_reconnect = 0

    def init_socket(self):
        if self.protocol == TCP:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)

    def connect(self):
        self.socket.connect((self.dest_ip,self.dest_port))
    def read_th(self):
        print('run read')
        header_size = self.constructor.header_0.size
        # wr_proc = mp.Process(target=self.write_th)
        # wr_proc.daemon = True
        # wr_proc.start()

        # self.socket.settimeout(10.0)
        while 1:

            data = self.socket.recv(header_size)
            if data:
                header, offset = self.constructor.parse_header(data,0)
                print(header.message_type)
                print(header.payload_size)
                if header.message_type == 200:
                    print('Ping')
                elif header.message_type == 201:
                    print('Pong')
                else:
                    received_payload_b = 0
                    payload = self.socket.recv(header.payload_size)
                    received_payload_b += len(payload)
                    while (received_payload_b < header.payload_size):
                        payload += self.socket.recv(header.payload_size - received_payload_b)
                        received_payload_b = len(payload)
                    print(f'try to decode message {header.message_type}')
                    self.decode_message_to_interface(header.message_type, payload, 0)


    def write_th(self):
        print('run write')
        tts = 1.0
        last = 0.0
        while 1:
            now = time.time()
            if (self.q_interface!=None):
                got, cmd = self.q_interface.get_cmd_to_dev()
                if got:
                    buffer = self.encode_cmd_to_buffer(cmd)
                    print('sending req')
                    if len(buffer) >0:
                        counter = 0
                        for i in range(5):
                            try:
                                self.socket.sendall(buffer)
                                break
                            except Exception as e:
                                counter+=1
                                print(f'Exception Сбой отправки TCP, повторная попытка {counter}, {e.args}')
                else:
                    time.sleep(0.01)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if (now - last) >= tts:
                print(f'elapsed {now-last}')
                msg = self.constructor.build_message_200()
                # new_im = im + random.randint(0, 100)
                # msg_3 = constructor.build_message_3(new_im, [detection])
                last = time.time()
                self.socket.sendall(msg)

    def rw_sim_work(self):

        if self.n_to_reconnect >0:
            connection_count = 0
        else:
            connection_count = -1
        while(connection_count<self.n_to_reconnect):
            if self.n_to_reconnect >0:
                connection_count+=1
            try:
                print('Client is running, v2')
                do_work = True

                self.init_socket()
                self.connect()
                th_read = thr.Thread(target=self.read_th)
                th_write = thr.Thread(target=self.write_th)
                th_read.daemon = True
                th_write.daemon = True
                th_write.start()
                th_read.start()

                while do_work:
                    do_work = th_read.is_alive() and th_write.is_alive()
                    time.sleep(2)
                print('client ruined')
            except Exception as e:
                print(e.args)
                print('Соединение разорвано, попытка переподключения через 2 секунды')
                time.sleep(2.0)
        return (False)


class V_net_server(V_net_entity):

    def __init__(self):
        super().__init__()

    def init_socket(self):
        if self.protocol == TCP:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.bind((self.self_ip, self.self_port))
            # self.socket.setblocking(True)

    def write(self):
        print('Write process started')
        run = True
        while run:
            # print('!!!!!!!!!!')
            got, cmd = self.q_interface.get_cmd_to_dev()
            if self.conn:
                if got:
                    if cmd:
                        buffer = self.encode_cmd_to_buffer(cmd)
                        if len(buffer)>0:
                            self.conn.sendall(buffer)
                else:
                    # print('QQQQQ Timed out')
                    time.sleep(0.05)
                    pass
            else:
                print('No Connection')
                run = False
        print('Write process finished')
    def run_v_server(self):
        # wr_th = thr.Thread(target=self.write)
        r_th = thr.Thread(target=self.work)
        # wr_th.daemon = True
        r_th.daemon = True

        r_th.run()
        # wr_th.run()
    def work(self):
        # self.run_write_thread()
        # TCP_IP = '127.0.0.1'
        #
        # TCP_PORT = 5005

        BUFFER_SIZE = self.constructor.header_0.size

        # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # s.bind((self.self_ip, self.self_port))
        constructor = self.constructor
        self.init_socket()
        while True:
            self.socket.listen(1)

            self.conn, addr = self.socket.accept()

            print('Connected')
            self.conn.settimeout(5)
            write_pr = mp.Process(target=self.write)
            write_pr.start()
                # write_pr.join()

            print('Connection address:', addr)
            flag = True
            try:
                while flag:

                    data = self.conn.recv(BUFFER_SIZE)

                    if len(data) > 0:
                        # print("received data:", data)
                        # q.put(f'received {len(data)} bytes')
                        header, offset = constructor.parse_header(data)
                        # header.print()
                        if header.message_type == 200:
                            self.conn.sendall(self.constructor.build_message_201())
                            print('200: Ping')
                        elif header.message_type == 201:
                            print('201: Pong')
                        # elif header.message_type ==51:
                        #     print('51: Frame req')
                        # elif header.message_type ==52:
                        #     print('52: Detections req')
                        # elif header.message_type ==53:
                        #     print('53: Frame and detections req')
                        else:
                            received_payload_b = 0
                            payload = self.conn.recv(header.payload_size)
                            received_payload_b += len(payload)
                            while (received_payload_b < header.payload_size):
                                payload += self.conn.recv(header.payload_size - received_payload_b)
                                received_payload_b = len(payload)
                            print(f'try to decode message {header.message_type}')
                            self.decode_message_to_interface(header.message_type,payload,0)
                            # if header.message_type == 3:
                            #     print('message 3 parsing')
                            #     meta, image, detections, offset = constructor.parse_message_3(payload, 0)
                            #     print(image.shape)
                            #     # res_im = (cv.resize(image, (500, 500)))
                            #     ima = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                            #     if self.q_interface != None:
                            #         self.q_interface.push_rep_from_dev(
                            #             Command('img_det', [meta, copy.deepcopy(ima), detections]))
                            #     print(len(detections))
                            #     print(detections[0].id)
                            #     print(detections[0].obj_class)
                            #     print(detections[0].bbox)

                    if not data:
                        flag = False
                if self.conn:
                    self.conn.close()
                    try:
                        write_pr.terminate()
                        print('connection closed')
                    except Exception as e:
                        print(e.args)
            except Exception as e:
                self.conn.close()
                try:
                    write_pr.terminate()
                    print('connection closed')
                except Exception as e:
                    print(e.args)
                print(e.args)
                # flag = False

# class V_net_client(V_net_entity):
#     def __init__(self):
#         super().__init__()
#
#     def init_socket(self):
#         self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)




def run_tcp_v_server(q_interface:Mp_dev_interface,TCP_IP = '127.0.0.1',TCP_PORT = 5005):
    # TCP_IP = '127.0.0.1'
    # TCP_PORT = 5005
    server = V_net_server()
    server.set_protocol(TCP)
    server.set_q_interface(q_interface)
    server.set_self_ipport(TCP_IP, TCP_PORT)
    server.run_v_server()
def run_tcp_client(q_interface:Mp_dev_interface,TCP_IP = '127.0.0.1',TCP_PORT = 5005,dest_ip='127.0.0.1', dest_port = 5006):
    print('start tcp client')
    client = V_net_client()
    client.set_q_interface(q_interface)
    client.set_protocol(TCP)
    client.set_self_ipport(TCP_IP,TCP_PORT)
    client.set_dest_ipport(dest_ip, dest_port)
    client.rw_sim_work()
def draw_detection(image:np.array,detection:p_cons.Detection,color = (0,255,0),thickness =0):
    cv.rectangle(image,detection.left_top(),detection.right_bottom(),color,thickness)
def run_show(q_interface:Mp_dev_interface):
    # window = cv.namedWindow('received',0)
    while True:
        got,cmd = q_interface.get_rep_from_dev()
        if got:
            if cmd.name == 'img_det':
                print('show')
                meta = cmd.value[0]
                meta.print()
                img = cmd.value[1]
                dets = cmd.value[2]
                for det in dets:
                    draw_detection(img,det)
                print(f'show: {img.shape}')
                # im0 = np.zeros((255,255), dtype=np.uint8)
                cv.imshow('w',img)
                cv.waitKey(1)
            else:
                # cmd.print()
                if cmd.name == CMD_list.get_det:
                    print(time.strftime('%H:%M:%S'), ' Получен запрос на обнаружения')
                    meta = cmd.value[0]
                    meta.print()

if __name__ == '__main__':
    interface = Mp_dev_interface()
    interface.allow_loose = True
    server_proc = mp.Process(target=run_tcp_v_server, args=(interface,'192.168.0.21',5005))
    # server_proc.daemon = True
    # show_proc = mp.Process(target=run_show, args=(interface,))
    server_proc.start()
    # # server_proc.join()
    # # run_show(interface)
    # show_proc.start()
    # run_tcp_v_server(None)
    # while True:
    #     got,cmd = interface.get_rep_from_dev()
    #     if got:
    #         cmd.print()
    # cl = Simple_udp_dialog(('localhost',0))
    # buf = cl.encode_ext_targeting(0,18883,55,30,1,2506,2250,1400)
    # id, params = cl.decode_message(buf)
    # print(id)
    # print(params)
    c = 0
    while 1:
        time.sleep(5)
        c+=5
        print(c)
