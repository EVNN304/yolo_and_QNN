from geometry_lib import Image_meta
import numpy as np
import cv2 as cv
import time
import copy
from toolset import Detection_centered, Codec_mini, Loc_polar_pt, Locator_track

JPEG = 1

class Detection:
    def __init__(self,bbox=[0,0,0,0],obj_class = 0,obj_p = 0.9,obj_id = 0):
        self.id = obj_id
        self.obj_class = obj_class
        self.p = obj_p
        self.bbox = bbox

    def left_top(self):
        return (self.bbox[0],self.bbox[1])

    def right_bottom(self):
        return (self.bbox[2], self.bbox[3])

    def width(self):
        return self.bbox[2]-self.bbox[0]

    def height(self):
        return self.bbox[3] - self.bbox[1]

    def print(self):
        print(f'Detection: bbox[{self.bbox}], class: {self.obj_class}, p: {self.p}, id:{self.id}')

    def get_int_center(self):
        xc = int((self.bbox[0]+self.bbox[2])/2)
        yc = int((self.bbox[1]+self.bbox[3])/2)
        return xc,yc

    def to_string(self):
        return f'Detection: bbox[{self.bbox}], class: {self.obj_class}, p: {self.p}, id:{self.id}'

    def to_string4table(self):
        return f'{self.obj_class};{self.p};{self.width()};{self.height()}'


def draw_detection(image:np.array,detection:Detection,color = (0,255,0),thickness =0):
    cv.rectangle(image,detection.left_top(),detection.right_bottom(),color,thickness)

class V_packet:
    def __init__(self, size = 0):
        self.size = size
        self.buffer = bytearray(size)

class V_header(V_packet):
    '''класс для работы с заголовком пакета'''
    def __init__(self,m_type: int = 0):
        super().__init__(5)
        self.size = 5
        self.message_type = m_type
        self.payload_size = 0
        self.set_message_type(m_type)
        self.timestamp = time.time()

    def print(self):
        overall = self.size + self.payload_size
        print(f'Header \n size = {self.size} bytes'
              f'\n message type: {self.message_type}'
              f'\n payload: {self.payload_size} bytes'
              f'\n overall size: {overall} bytes ({(overall)/1024} kb)')

    def set_message_type(self, m_type:int):
        self.message_type = m_type
        self.buffer[0:1] = m_type.to_bytes(1,byteorder='little',signed=False)

    def set_payload_size(self,payload_size:int):
        self.payload_size = payload_size
        self.buffer[1:] = payload_size.to_bytes(4,byteorder='little',signed=False)

    def parse_from_bytes(self,buffer:bytearray, offset:int = 0,update_self = True):

        m_type = int.from_bytes(buffer[offset:offset+1],byteorder='little',signed=False)
        payload_size = int.from_bytes(buffer[offset+1:offset+5],byteorder='little',signed=False)

        if update_self:
            self.message_type = m_type
            self.payload_size = payload_size
        return m_type,payload_size

    def set_timestamp(self,time):
        self.timestamp = time

    def get_bytes(self):
        return self.buffer


class V_packet_0_body(V_packet):
    '''
    Реализация конструктора пакета [0] - для передачи изображения без отметок и сжатия
    '''
    def __init__(self):
        super().__init__()

    # def to_buf(self):
    #

class V_constructor:
    '''
    Конструктор пакетов
    '''

    def __init__(self):
        self.header_0 = V_header()
        self.b_order = 'little'
        # self.frame_description_bsize = 18 #в предыдущей версии протокола
        self.frame_description_bsize = 26 #05.07.2022 в описание добавлена метка времени
        self.compression_alg = JPEG
        self.compression_q = 100 #качество сжатого изображения
        self.default_meta = Image_meta()
        self.detection_style = 0
        self.DET_STYLE_CVT = 12
        self.DET_STYLE_BBOX = 0
        self.DET_STYLE_CBOX = 1
        self.DET_STYLE_NEURO_DET = 3
        self.mini_codec = Codec_mini()
        self.byte_flag_masks = (b'\01'[0],b'\02'[0],b'\x04'[0],b'\x08'[0],b'\x10'[0],b'\x20'[0],b'\x40'[0],b'\x80'[0])
    def build_message_0(self,img,meta = None):
        '''
        Формат пакета [0]: несжатое изображение без отметок
        :param img:
        :param meta:
        :return:
        '''
        if meta == None:
            meta = self.default_meta
            stamp = time.time()
        else:
            stamp = meta.timestamp
        self.header_0.set_message_type(0)
        self.header_0.set_timestamp(stamp)
        payload = self.encode_raw_image(img,meta)
        self.header_0.set_payload_size(len(payload))
        return self.header_0.get_bytes()+payload

    def parse_message_0(self,buffer,offset = 0):
        return self.decode_raw_image(buffer,offset)

    def build_message_1(self,img,meta = None):
        self.header_0.set_message_type(1)
        if meta == None:
            meta = self.default_meta
        payload = self.encode_compressed_image(img, meta,
                                               c_alg=self.compression_alg,
                                               quality=self.compression_q,
                                               auto_meta=True)
        self.header_0.set_payload_size(len(payload))
        print('Размер пакета ',self.header_0.payload_size)
        return self.header_0.get_bytes() + payload

    def parse_message_1(self,buffer,offset):
        return self.decode_compressed_image(buffer,offset)



    def build_message_2(self,detections,meta = None):
        self.header_0.set_message_type(2)
        if meta == None:
            meta = self.default_meta
        payload = self.encode_frame_description(meta)+self.encode_detections(detections,self.detection_style)
        self.header_0.set_payload_size(len(payload))
        return self.header_0.get_bytes()+payload

    def parse_message_2(self,buffer,offset):
        meta, offset =self.decode_frame_description(buffer,offset)
        detections, offset = self.decode_detections(buffer,offset)
        return meta, detections,offset

    def build_message_3(self,img,detections,meta = None):
        self.header_0.set_message_type(3)
        if meta == None:
            meta = self.default_meta
        payload_img = self.encode_compressed_image(img,meta,self.compression_alg,self.compression_q,False)
        payload_dets = self.encode_detections(detections,self.detection_style)
        self.header_0.set_payload_size(len(payload_dets)+len(payload_img))
        return self.header_0.get_bytes()+payload_img+payload_dets

    def parse_message_3(self,buffer,offset):
        meta, image, offset = self.decode_compressed_image(buffer,offset)
        detections, offset = self.decode_detections(buffer,offset)
        return meta,image,detections,offset

    def build_message_4(self,encoded_tracks,meta):
        self.header_0.set_message_type(4)
        payload = bytearray(0)

        if meta == None:
            meta = self.default_meta
        payload += self.encode_frame_description(meta)
        tracks_n = len(encoded_tracks)
        payload += int(tracks_n).to_bytes(2,self.b_order,signed = False)
        for encoded_track in encoded_tracks:
            payload+=encoded_track
        self.header_0.set_payload_size(len(payload))
        return self.header_0.get_bytes()+payload

    def parse_message_4(self,buffer,offset):
        meta, offset = self.decode_frame_description(buffer, offset)
        tracks_n = int.from_bytes(buffer[offset:offset+2],self.b_order)
        offset+=2
        decoded_simple_tracks = []
        for i in range(tracks_n):
            new_track,offset = self.mini_codec.decode_track_to_simple_track(buffer,offset)
            decoded_simple_tracks.append(copy.copy(new_track))
        return meta,decoded_simple_tracks, offset

    def build_message_5(self,detections,imgs,meta = None):
        if str(type(imgs))=="<class 'list'>":
            images_list = imgs
        elif str(type(imgs)) == "<class 'numpy.ndarray'>":
            images_list = []
            for det in detections:
                x1,y1 = det.left_top()
                x2,y2 = det.right_bottom()
                images_list.append(imgs[y1:y2,x1:x2])
                # cv.imshow('wwwwwwwi',imgs[y1:y2,x1:x2])
                # cv.waitKey()
        else:
            imgs = np.zeros((4054,4504),dtype=np.uint8)
            images_list = []
            for det in detections:
                x1, y1 = det.left_top()
                x2, y2 = det.right_bottom()
                images_list.append(imgs[y1:y2, x1:x2])
        self.header_0.set_message_type(5)
        if meta == None:
            meta = self.default_meta
        payload = b''
        payload+=self.encode_frame_description(meta)
        payload+=self.encode_dets_header(len(detections),self.detection_style)
        for i,det in enumerate(detections):
            # print('in ', i)
            payload+=self.encode_single_detection(det,self.detection_style)[0]
            payload+=self.encode_single_image(images_list[i],self.compression_q)

        self.header_0.set_payload_size(len(payload))
        return self.header_0.get_bytes()+payload

    def parse_message_5(self,buffer,offset):
        meta, offset = self.decode_frame_description(buffer, offset)
        n_dets,det_style,offset = self.decode_dets_header(buffer,offset)
        dets = []
        images = []
        for i in range(n_dets):
            det,offset = self.decode_single_detection(buffer,offset,det_style)
            dets.append(copy.deepcopy(det))
            image,offset = self.decode_single_image(buffer,offset)
            images.append(image)
        return meta,dets,images,offset

    def build_message_60(self,stamp = 0.0,mode:int = 0, reserved = None):
        '''
        Сборка сообщения для управления режимами работы сервера
        '''
        self.header_0.set_message_type(60)
        payload = b''
        if stamp > 0:
            enc_stamp = stamp
        else:
            enc_stamp = time.time()
        payload += int(enc_stamp * 1000).to_bytes(8, self.b_order, signed=False)
        payload += int(mode).to_bytes(1, self.b_order, signed=False)
        for i in range(4):
            res = int(0).to_bytes(2, self.b_order, signed=True)
            try:
                if reserved!=None:
                    res = int(reserved[i]).to_bytes(2,self.b_order,signed=True)
            except:
                pass
            payload+=res
        self.header_0.set_payload_size(len(payload))
        return self.header_0.get_bytes() + payload

    def parse_message_60(self,buffer,offset):
        '''
        Разбор
        '''
        stamp = int.from_bytes(buffer[offset:offset + 8], self.b_order, signed=False)/1000
        offset += 8
        mode = int.from_bytes(buffer[offset:offset + 1], self.b_order, signed=False)
        offset += 1
        reserved = [0,0,0,0]
        for i in range(len(reserved)):
            reserved[i] = int.from_bytes(buffer[offset:offset + 2], self.b_order, signed=True)
            offset += 2
        return stamp, mode,reserved, offset

    def build_message_71(self,stamp = 0.0,az2go = 0.0, el2go = 0.0, reserved = None):
        '''
        Сборка сообщения - команды для поворота в заданную точку
        '''
        self.header_0.set_message_type(71)
        payload = b''
        if stamp>0:
            enc_stamp = stamp
        else:
            enc_stamp = time.time()
        payload+=int(enc_stamp*1000).to_bytes(8,self.b_order,signed=False)
        payload+=int((az2go%360)*(65535/360)).to_bytes(2,self.b_order,signed=False)
        payload+=int((el2go)*(32765/90)).to_bytes(2,self.b_order,signed=True)
        for i in range(4):
            res = int(0).to_bytes(2, self.b_order, signed=True)
            try:
                if reserved!=None:
                    res = int(reserved[i]).to_bytes(2,self.b_order,signed=True)
            except:
                pass
            payload+=res
        self.header_0.set_payload_size(len(payload))
        return self.header_0.get_bytes() + payload

    def parse_message_71(self,buffer,offset):
        '''
        Разбор
        '''
        stamp = int.from_bytes(buffer[offset:offset + 8], self.b_order, signed=False)/1000
        offset += 8
        az2go = round(int.from_bytes(buffer[offset:offset + 2], self.b_order, signed=False)*360/65535,2)
        offset += 2
        el2go = round(int.from_bytes(buffer[offset:offset + 2], self.b_order, signed=True)*90/32765,2)
        offset += 2
        reserved = [0,0,0,0]
        for i in range(len(reserved)):
            reserved[i] = int.from_bytes(buffer[offset:offset + 2], self.b_order, signed=True)
            offset += 2
        return stamp, (az2go,el2go),reserved, offset


    def parse_locator_track_header(self,buffer,offset):
        '''
        Разбор заголовка траектории от локатора из сообщения "101"
        '''
        track_id = int.from_bytes(buffer[offset:offset+4],'little',signed=False)
        track_id-=1
        points_count = int.from_bytes(buffer[offset+4:offset+5],'little',signed=False)
        offset+=5
        return track_id,points_count,offset

    def parse_locator_track_header_v2(self,buffer,offset):
        '''
        Разбор заголовка траектории от локатора из сообщения "110"
        '''
        track_id = int.from_bytes(buffer[offset:offset+4],'little',signed=False)
        offset+=4
        track_id-=1
        dang_flag = int.from_bytes(buffer[offset:offset+1],'little',signed=False)
        offset+=1
        obj_type = int.from_bytes(buffer[offset:offset+1],'little',signed=False)
        offset+=1
        points_count = int.from_bytes(buffer[offset:offset+1],'little',signed=False)
        offset+=1

        # print(f'track: {track_id}, dang: {dang_flag}, type: {obj_type}; n_points: {points_count}')
        return track_id,dang_flag, obj_type, points_count,offset

    def build_locator_track_header_v2(self,track:Locator_track,max_points = 10):
        header_b = b''
        header_b+=int(track.id+1).to_bytes(4,'little',signed = False)
        header_b+=int(track.dang_flag).to_bytes(1,'little',signed=False)
        header_b+=int(track.obj_type).to_bytes(1,'little',signed=False)
        points_count = min(len(track.pts_list),max_points)
        header_b+=int(points_count).to_bytes(1,'little',signed=False)
        return header_b



    def parse_locator_single_point(self,buffer,offset):
        timestamp = int.from_bytes(buffer[offset:offset+4],'little',signed=False)/1000
        offset+=4
        az = int.from_bytes(buffer[offset:offset+2],'little', signed=False)/100
        offset+=2
        el = int.from_bytes(buffer[offset:offset+2], 'little', signed=False)/100 - 90.0
        offset += 2
        r = int.from_bytes(buffer[offset:offset + 4], 'little', signed=False) / 100
        offset += 4
        v_r = int.from_bytes(buffer[offset:offset + 2], 'little', signed=True) / 100
        offset += 2
        locator_pt = Loc_polar_pt(timestamp,az,el,r,v_r)

        return locator_pt, offset

    def parse_locator_single_point_v2(self,buffer,offset,ref_stamp = 0.0):

        timestamp = ref_stamp + int.from_bytes(buffer[offset:offset+4],'little',signed=True)/1000
        offset+=4
        az = int.from_bytes(buffer[offset:offset+2],'little', signed=False)/100
        offset+=2
        el = int.from_bytes(buffer[offset:offset+2], 'little', signed=False)/100 -90.0
        offset += 2
        r = int.from_bytes(buffer[offset:offset + 4], 'little', signed=False) / 100
        offset += 4
        v_r = int.from_bytes(buffer[offset:offset + 2], 'little', signed=True) / 100
        offset += 2
        locator_pt = Loc_polar_pt(timestamp,az,el,r,v_r)
        # print(f'____pt {timestamp}, az: {az}, el: {el}, r: {r}, vr: {v_r}')
        return locator_pt, offset

    def build_locator_single_point_v2(self,loc_point:Loc_polar_pt,ref_stamp = time.time()):
        single_pt_b = b''
        single_pt_b+= int((loc_point.timestamp-ref_stamp)*1000).to_bytes(4,'little',signed=True)
        single_pt_b+= int((loc_point.az%360)*100).to_bytes(2,'little', signed=False)
        single_pt_b += int((loc_point.el+90) * 100).to_bytes(2, 'little', signed=False)
        single_pt_b += int(loc_point.r * 100).to_bytes(4,'little', signed=False)
        single_pt_b += int(loc_point.vr * 100 ).to_bytes(2, 'little', signed=True)
        return single_pt_b


    def parse_message_101(self,buffer,offset):
        '''
        Разбор сообщения с траекториями от локатора
        '''
        # print('Decoding tracks from locator....')
        tracks_count = int.from_bytes(buffer[offset:offset+2],'little',signed=False)
        offset+=2
        # print(tracks_count)
        tracks_list = []
        for tr_i in range(tracks_count):
            tr_id,pts_count,offset = self.parse_locator_track_header(buffer,offset)
            tr = Locator_track(tr_id)
            # print(f'first_track: id = {tr_id}, pts_n = {pts_count}')
            for i in range(pts_count):
                pt,offset = self.parse_locator_single_point(buffer,offset)
                tr.add_pt(pt)
                # pt.print()
            tracks_list.append(tr)
        # for tr in tracks_list:
        #     tr.print()
        check_sum = buffer[offset:offset+1]
        # print(f'Check sum: {check_sum}, {int.from_bytes(check_sum,"little",signed=False)}')
        offset +=1
        return tracks_list,offset

    def parse_message_102(self,buffer,offset):
        '''
        разбор сообщения с текущим углом поворота ОПУ
        '''
        # print('decoding locator direction')
        current_az = int.from_bytes(buffer[offset:offset + 2], 'little', signed=False) / 100
        offset += 2
        opu_t = int.from_bytes(buffer[offset:offset + 2], 'little', signed=False) / 100
        offset += 2
        # print(f'------------------------------------------------------------------->>{current_az},{opu_t}')
        return current_az,opu_t,offset

    def parse_message_104(self,buffer,offset):
        '''
        Разбор сообщения с текущим углом поворота и статусом РЛС
        '''
        # print('decoding locator stats')
        current_az = int.from_bytes(buffer[offset:offset + 2], 'little', signed=False) / 100
        offset += 2
        opu_t = int.from_bytes(buffer[offset:offset + 2], 'little', signed=False) / 100
        offset += 2
        opu_stat = int.from_bytes(buffer[offset:offset + 1], 'little', signed=False)
        offset+=1
        signal_proc_stat = int.from_bytes(buffer[offset:offset + 1], 'little', signed=False)
        offset += 1
        reserv_1 = int.from_bytes(buffer[offset:offset + 2], 'little', signed=False)
        offset += 2
        reserv_2 = int.from_bytes(buffer[offset:offset + 2], 'little', signed=False)
        offset += 2
        return [current_az, opu_t, opu_stat, signal_proc_stat, reserv_1, reserv_2], offset

    def parse_massage_110(self,buffer,offset):
        '''
                Разбор сообщения с траекториями от локатора, обновленная версия
        '''
        # print('Decoding tracks from locator....NEW')
        ref_timestamp = int.from_bytes(buffer[offset:offset+8],'little', signed=False)/1000
        # print(f'ref_timestamp:{ref_timestamp} -- {time.time()}')
        offset+=8
        tracks_count = int.from_bytes(buffer[offset:offset + 2], 'little', signed=False)
        # print(f'tracks count: {tracks_count}')
        offset += 2
        # print(tracks_count)
        tracks_list = []
        for tr_i in range(tracks_count):
            tr_id,dang_flag,obj_type, pts_count, offset = self.parse_locator_track_header_v2(buffer, offset)
            tr = Locator_track(tr_id)
            tr.dang_flag = dang_flag
            tr.obj_type = obj_type
            # print(f'first_track: id = {tr_id}, pts_n = {pts_count}')
            for i in range(pts_count):
                pt, offset = self.parse_locator_single_point_v2(buffer, offset,ref_timestamp)
                tr.add_pt(pt)
                # pt.print()
            tracks_list.append(tr)
        # for tr in tracks_list:
        #     tr.print()
        check_sum = buffer[offset:offset + 1]
        # print(f'Check sum: {check_sum}, {int.from_bytes(check_sum, "little", signed=False)}')
        offset += 1
        return tracks_list, offset

    def build_message_110(self,tracks_list,ref_stamp = time.time()):
        max_points = 5
        '''сборка сообщения с траекториями от локатора'''
        # print('encoding tracks from locator....NEW')
        payload =b''
        payload+=int(ref_stamp*1000).to_bytes(8,'little',signed = False)
        tracks_count = len(tracks_list)
        payload+=int(tracks_count).to_bytes(2, 'little', signed = False)
        # tracks_count = int.from_bytes(buffer[offset:offset + 2], 'little', signed=False)
        # print(f'tracks count: {tracks_count}')
        for tr in tracks_list:
            pts_count = min(len(tr.pts_list), max_points)
            payload+= self.build_locator_track_header_v2(tr,pts_count)
            for i in range(pts_count):
                pt = tr.pts_list[len(tr.pts_list)-pts_count+i]
                payload+= self.build_locator_single_point_v2(pt,ref_stamp)
        self.header_0.set_message_type(110)
        self.header_0.set_payload_size(len(payload))
        return self.header_0.get_bytes()+payload





    def build_message_103(self,az,el):
        '''
        Сообщение с направлением камеры
        '''
        self.header_0.set_message_type(103)
        payload = bytearray(4)

        payload[0:2] = int(az*100).to_bytes(2, self.b_order, signed=False)
        payload[2:4] = int((el+90)*100).to_bytes(2, self.b_order, signed=False)
        self.header_0.set_payload_size(len(payload))
        return self.header_0.get_bytes()+payload

    def build_message_105(self,control_params):
        '''
        Сообщение для управления РЛС
        '''
        # print(f'control params : {control_params}')
        self.header_0.set_message_type(105)
        payload = bytearray(7)
        payload[0:1] = int(control_params[0]).to_bytes(1, self.b_order, signed=False)
        payload[1:3] = int(control_params[1]).to_bytes(2, self.b_order, signed=False)
        payload[3:5] = int(control_params[2]).to_bytes(2, self.b_order, signed=False)
        payload[5:7] = int(control_params[3]).to_bytes(2, self.b_order, signed=False)
        self.header_0.set_payload_size(len(payload))
        # print('payload: ',self.header_0.get_bytes() + payload)
        return self.header_0.get_bytes()+payload

    def build_message_61(self,flags,vals):
        '''
        Сообщение для настройки камеры. Флаги [авто эксп, авто усил.], значения [экспозиция, усиление, целевая яркость]
        :return:
        '''

        self.header_0.set_message_type(61)
        payload = bytearray(16)
        #выставляем флаги
        # print('выставляем флаги')
        offset = 0
        for i,flag in enumerate(flags):
            # print(i,'>',flag)
            if flag:
                payload[offset]|=self.byte_flag_masks[i]
        offset+=1
        payload[offset:offset+4] = int(vals[0]*100).to_bytes(4,self.b_order,signed=False)
        offset+=4
        payload[offset:offset+2] = int(vals[1]*10).to_bytes(2,self.b_order,signed=True)
        offset+=2
        payload[offset:offset+1] = int(vals[2]).to_bytes(1,self.b_order,signed=False)
        offset+=1

        self.header_0.set_payload_size(len(payload))
        return self.header_0.get_bytes() + payload

    def parse_message_61(self,buffer,offset):
        flags = [True]*8
        vals = []
        #Распаковываем флаги
        # print('Распаковываем флаги')
        # print(buffer[offset:offset + 1])
        # print(buffer[offset:offset + 1][0])
        for i,flag in enumerate(flags):
            flags[i] = (buffer[offset:offset+1][0]&self.byte_flag_masks[i])>0
        offset+=1
        #Распаковываем значения параметров
        expo_val = int.from_bytes(buffer[offset:offset+4],self.b_order,signed=False)/100
        vals.append(expo_val)
        offset+=4
        gain_val = int.from_bytes(buffer[offset:offset+2],self.b_order,signed=True)/10
        vals.append(gain_val)
        offset+=2
        target_brightness = int.from_bytes(buffer[offset:offset+1],self.b_order,signed=False)
        vals.append(target_brightness)
        offset+=1
        reserv_1 = int.from_bytes(buffer[offset:offset+2],self.b_order,signed=True)
        vals.append(reserv_1)
        offset += 2
        reserv_2 = int.from_bytes(buffer[offset:offset + 2], self.b_order, signed=True)
        vals.append(reserv_2)
        offset += 2
        reserv_3 = int.from_bytes(buffer[offset:offset + 2], self.b_order, signed=True)
        vals.append(reserv_3)
        offset += 2
        reserv_4 = int.from_bytes(buffer[offset:offset + 2], self.b_order, signed=True)
        vals.append(reserv_4)
        offset += 2
        return flags, vals, offset


    def build_message_62(self, vals):
        '''
                Сообщение для настройки обработчика. значения [порог бинаризации, размер ядра эрозии, размер ядра дилатации]
                :return:
        '''
        self.header_0.set_message_type(62)
        payload = bytearray(12)
        offset = 0
        payload[offset:offset + 1] = int(vals[0]).to_bytes(1, self.b_order, signed=False)
        offset += 1
        payload[offset:offset + 1] = int(vals[1]).to_bytes(1, self.b_order, signed=False)
        offset += 1
        payload[offset:offset + 1] = int(vals[2]).to_bytes(1, self.b_order, signed=False)
        offset += 1

        self.header_0.set_payload_size(len(payload))
        return self.header_0.get_bytes() + payload

    def parse_message_62(self,buffer,offset):
        vals = []
        # Распаковываем значения параметров
        bit_thr = int.from_bytes(buffer[offset:offset + 1], self.b_order, signed=False)
        vals.append(bit_thr)
        offset += 1
        er_k_size = int.from_bytes(buffer[offset:offset + 1], self.b_order, signed=False)
        vals.append(er_k_size)
        offset += 1
        dil_k_size = int.from_bytes(buffer[offset:offset + 1], self.b_order, signed=False)
        vals.append(dil_k_size)
        offset += 1
        reserv_1 = int.from_bytes(buffer[offset:offset + 2], self.b_order, signed=True)
        vals.append(reserv_1)
        offset += 2
        reserv_2 = int.from_bytes(buffer[offset:offset + 2], self.b_order, signed=True)
        vals.append(reserv_2)
        offset += 2
        reserv_3 = int.from_bytes(buffer[offset:offset + 2], self.b_order, signed=True)
        vals.append(reserv_3)
        offset += 2
        reserv_4 = int.from_bytes(buffer[offset:offset + 2], self.b_order, signed=True)
        vals.append(reserv_4)
        offset += 2
        return vals, offset


    def build_message_131(self,detections,meta,style = None):
        '''
        Сообщение с сырыми обнаружениями
        :return:
        '''
        if style == None:
            style = self.DET_STYLE_CVT
        self.header_0.set_message_type(131)
        payload = self.encode_frame_description(meta)
        payload+= self.encode_detections(detections,self.DET_STYLE_CVT,style)
        self.header_0.set_payload_size(len(payload))
        return self.header_0.get_bytes()+payload

    def parse_message_131(self,buffer,offset):
        meta, offset = self.decode_frame_description(buffer,offset)
        cvt_boxes, offset = self.decode_detections(buffer,offset)

        return meta, cvt_boxes, offset

    def build_message_200(self):
        '''
        Ping запрос на проверку связи
        '''
        self.header_0.set_message_type(200)
        self.header_0.set_payload_size(0)
        return self.header_0.get_bytes()

    def build_message_201(self):
        '''
        Pong ответ на проверку связи
        '''
        self.header_0.set_message_type(201)
        self.header_0.set_payload_size(0)
        return self.header_0.get_bytes()

    def build_meta_req(self,meta:Image_meta,type = 53):
        self.header_0.set_message_type(type)
        self.header_0.set_payload_size(self.frame_description_bsize)
        payload = self.encode_frame_description(meta)
        return self.header_0.get_bytes()+payload


    def parse_meta_req(self,buffer,offset):
        meta,offset = self.decode_frame_description(buffer,offset)
        return meta,offset

    def build_roi_req(self,meta:Image_meta, areas,type = 54):
        '''
        Собрать сообщение - запрос детектирования в выделенных областях
        Тип сообщения 54 - без изображения
        Тип сообщения 55 - с изображением
        '''
        self.header_0.set_message_type(type)
        roi_b_packs = len(areas).to_bytes(2, self.b_order, signed=False)
        for area in areas:
            roi_b_packs+=self.encode_roi(area)
        payload = self.encode_frame_description(meta)+roi_b_packs
        self.header_0.set_payload_size(len(payload))
        return self.header_0.get_bytes()+payload

    def parse_roi_req(self,buffer,offset):
        meta, offset = self.decode_frame_description(buffer, offset)
        areas_n = int.from_bytes(buffer[offset:offset+2],self.b_order,signed=False)
        areas = []
        offset+=2
        for i in range(areas_n):
            area,offset = self.decode_roi(buffer,offset)
            areas.append(area)
        return meta,areas,offset

    def build_t_recognition_req(self,stamp,global_targets:[],type = 56):
        '''
        Собрать сообщение - запрос на распознавание по целеуказаниям
        :param stamp: Метка времени, с
        :param global_targets: массив целеуказаний [[id,az,el],.....]
        :param type: Тип сообщения (56 - запрос на распознавания без изображения)
        :return:
        '''
        self.header_0.set_message_type(type)
        payload=int(stamp * 1000000).to_bytes(8, self.b_order, signed=False)
        payload += len(global_targets).to_bytes(2, self.b_order, signed=False)
        for target in global_targets:
            id = target[0]
            payload += id.to_bytes(2, self.b_order, signed=False)
            az = target[1]
            payload += int(az*1000).to_bytes(3,self.b_order,signed=True)
            el = target[2]
            payload += int(el * 1000).to_bytes(3, self.b_order, signed=True)
        self.header_0.set_payload_size(len(payload))
        return self.header_0.get_bytes() + payload

    def parse_t_recognition_req(self,buffer,offset):
        timestamp = int.from_bytes(buffer[offset:offset+8],self.b_order,signed=False)/1000000
        offset+=8
        areas_n = int.from_bytes(buffer[offset:offset + 2], self.b_order, signed=False)
        offset+=2
        targetings = []
        if areas_n>0:
            for i in range(areas_n):
                id = int.from_bytes(buffer[offset:offset + 2], self.b_order, signed=False)
                offset+=2
                az = int.from_bytes(buffer[offset:offset + 3], self.b_order, signed=True)/1000
                offset+=3
                el = int.from_bytes(buffer[offset:offset + 3], self.b_order, signed=True) / 1000
                offset += 3
                targetings.append([id,az,el])
        return timestamp,targetings,offset






    def parse_header(self,buffer,offset = 0):
        header = V_header()
        header_bytes = buffer[offset:offset+header.size]
        header.parse_from_bytes(header_bytes)
        # header.print()
        offset+=header.size
        return header, offset

    def encode_roi(self,roi):
        '''

        :param meta: Выделяемая область:[x1,y1,x2,y2]
        :return: байты для вставки в пакет
        '''
        roi_bytes = bytearray(8)
        roi_bytes[0:2] = roi[0].to_bytes(2, self.b_order, signed=False)
        roi_bytes[2:4] = roi[1].to_bytes(2, self.b_order, signed=False)
        roi_bytes[4:6] = roi[2].to_bytes(2, self.b_order, signed=False)
        roi_bytes[6:8] = roi[3].to_bytes(2, self.b_order, signed=False)
        return roi_bytes

    def decode_roi(self,buffer,offset):
        roi = [0,0,0,0]
        for i in range(len(roi)):
            roi[i]= int.from_bytes(buffer[offset:offset + 2], self.b_order, signed=False)
            offset+=2
        return roi, offset


    def encode_frame_description(self,meta:Image_meta):
        '''
        id - 1 байт, беззнаковое целое
        w_px - 2 байта, беззнаковое целое
        h_px - 2 байта, беззнаковое целое
        channel_count - 2 байта, беззнаковое целое
        az, el - 3 байта, знаковое целое, масштабирующий коэфф. 0.001
        w_deg,h_deg - 2 байта, беззнаковое целое, масштабирующий коэфф. 0.01
        общий размер 26 байт
        :param meta: Описание кадра
        :return: байты для вставки в пакет
        '''
        # b_order = 'little'
        meta_bytes = bytearray(self.frame_description_bsize)
        im_w = meta.im_size[0]
        im_h = meta.im_size[1]
        w_deg = meta.angle_s[0]
        h_deg = meta.angle_s[1]
        meta_bytes[0:8] = int(meta.timestamp*1000000).to_bytes(8,self.b_order,signed=False)
        meta_bytes[8:10] = meta.id.to_bytes(2,self.b_order,signed=False)
        meta_bytes[10:12] = im_w.to_bytes(2,self.b_order,signed=False)
        meta_bytes[12:14] = im_h.to_bytes(2,self.b_order,signed=False)
        meta_bytes[14:16] = meta.channel_count.to_bytes(2, self.b_order, signed=False)
        meta_bytes[16:19] = int(meta.az*1000).to_bytes(3,self.b_order,signed=True)
        meta_bytes[19:22] = int(meta.el * 1000).to_bytes(3, self.b_order, signed=True)
        meta_bytes[22:24] = int(w_deg*100).to_bytes(2,self.b_order,signed=False)
        meta_bytes[24:26] = int(h_deg * 100).to_bytes(2, self.b_order, signed=False)
        # meta_bytes[0:2] = meta.id.to_bytes(2,self.b_order,signed=False)
        # meta_bytes[2:4] = im_w.to_bytes(2,self.b_order,signed=False)
        # meta_bytes[4:6] = im_h.to_bytes(2,self.b_order,signed=False)
        # meta_bytes[6:8] = meta.channel_count.to_bytes(2, self.b_order, signed=False)
        # meta_bytes[8:11] = int(meta.az*1000).to_bytes(3,self.b_order,signed=True)
        # meta_bytes[11:14] = int(meta.el * 1000).to_bytes(3, self.b_order, signed=True)
        # meta_bytes[14:16] = int(w_deg*100).to_bytes(2,self.b_order,signed=False)
        # meta_bytes[16:18] = int(h_deg * 100).to_bytes(2, self.b_order, signed=False)
        return meta_bytes

    def decode_frame_description(self,packet:bytearray, offset = 0, meta = None):
        '''
        Декодирование описания изображения из пакета. (см. encode_frame_discription)
        :param packet: Полный пакет или фрагмент с параметрами кадра
        :param offset: Отступ от начала пакета в байтах
        :param meta: Описание для редактирования. Если нет, будет создано новое
        :return:
        '''
        if meta == None:
            meta = Image_meta()
        meta.timestamp = int.from_bytes(packet[offset:offset+8],self.b_order,signed=False)/1000000
        meta.id = int.from_bytes(packet[offset+8:offset+10],self.b_order,signed=False)
        im_w = int.from_bytes(packet[offset+10:offset+12],self.b_order,signed=False)
        im_h = int.from_bytes(packet[offset+12:offset+14], self.b_order, signed=False)
        meta.channel_count = int.from_bytes(packet[offset+14:offset+16], self.b_order, signed=False)
        meta.az = 0.001*int.from_bytes(packet[offset+16:offset+19],self.b_order,signed=True)
        meta.el = 0.001 * int.from_bytes(packet[offset + 19:offset + 22], self.b_order, signed=True)
        w_deg = 0.01 * int.from_bytes(packet[offset + 22:offset + 24], self.b_order, signed=False)
        h_deg = 0.01 * int.from_bytes(packet[offset + 24:offset + 26], self.b_order, signed=False)
        meta.set_sizes([im_w,im_h],[w_deg,h_deg])
        offset+=self.frame_description_bsize
        return meta, offset

    def encode_raw_image(self,image:np.array, meta:Image_meta = None, auto_meta = True):
        '''

        :param image: входное изображение
        :param meta: параметры изображения. Если не заданы, будут сконфигурированы автоматически
        :param auto_meta: автоматическая корректировка параметров
        :return:
        '''
        if meta == None:
            meta = Image_meta()
            auto_meta = True
        if auto_meta:
            meta_changed, meta = self.adapt_meta(image.shape)
        im_b_size = meta.channel_count*meta.im_size[0]*meta.im_size[1]
        b_pack = bytearray(self.frame_description_bsize+im_b_size)
        b_pack[0:self.frame_description_bsize] = self.encode_frame_description(meta)
        b_pack[self.frame_description_bsize:] = image.tobytes()
        return b_pack

    def decode_raw_image(self,buffer,offset):
        meta, h_offset = self.decode_frame_description(buffer,offset)
        img_size = int(meta.im_size[0]) * int(meta.im_size[1]) * int(meta.channel_count)
        if meta.channel_count == 1:
            img_shape = (meta.im_size[1], meta.im_size[0])
        else:
            img_shape = (meta.im_size[1], meta.im_size[0], meta.channel_count)
        # print(f'restored img_shape: {img_shape}')
        img = np.frombuffer(buffer, dtype=np.uint8, count=img_size, offset=self.frame_description_bsize + offset).reshape(
            img_shape)
        offset+=self.frame_description_bsize
        offset+=img_size
        return meta,img, offset

    def encode_compressed_image(self,image:np.array, meta:Image_meta = None, c_alg = JPEG, quality = 100, auto_meta = False):
        '''
        Упаковывает сжатое изображение в пакет байтов
        :param image: входное изображение
        :param meta: параметры изображения
        :param c_alg: алгоритм сжатия
        :param quality: качество
        :param auto_meta:
        :return:
        '''
        # meta.print()
        # print(f'auto meta = {auto_meta}')
        if meta == None:
            meta = Image_meta()
            auto_meta = True
            # print('set auto meta')
        if auto_meta:
            meta_changed, meta = self.adapt_meta(image.shape,meta)
        img_in_bytes = bytearray()
        if c_alg == JPEG:
            encode_param = [int(cv.IMWRITE_JPEG_QUALITY), quality]
            result, imgencode = cv.imencode(".jpg", image, encode_param)
            img_in_bytes = imgencode.tobytes()
            # print('img size: ',len(img_in_bytes))
        # print('in encode')
        # meta.print()
        meta_b = self.encode_frame_description(meta)
        compressed_header = bytearray(7)
        compressed_header[0:2] = int(c_alg).to_bytes(2,self.b_order,signed=False)
        compressed_header[2:4] = int(quality).to_bytes(2, self.b_order, signed=False)
        compressed_header[4:7] = len(img_in_bytes).to_bytes(3, self.b_order, signed=False)
        return meta_b + compressed_header+img_in_bytes

    def decode_compressed_image(self,buffer,offset=0):
        meta, h_offset = self.decode_frame_description(buffer[offset:offset+self.frame_description_bsize])
        offset+=self.frame_description_bsize
        c_alg = int.from_bytes(buffer[offset:offset+2],self.b_order,signed=False)
        quality = int.from_bytes(buffer[offset+2:offset+4],self.b_order,signed=False)
        img_b_size = int.from_bytes(buffer[offset+4:offset + 7], self.b_order, signed=False)
        img_from_bytes = cv.imdecode(np.frombuffer(buffer[offset+7:offset+7+img_b_size],
                                                   dtype=np.uint8), cv.IMWRITE_JPEG_QUALITY)
        offset+=img_b_size
        offset+=7
        return meta, img_from_bytes, offset

    def adapt_meta(self,image_shape, meta:Image_meta = None):
        '''
        Функция для проверки параметров изображения и исправления при необходимости
        :param image_shape:
        :param meta:
        :return:
        '''
        meta_changed = False

        if meta == None:
            meta_changed = True
            meta = Image_meta()
        if len(image_shape) ==2:
            meta.channel_count = 1
        else:
            if meta.channel_count != image_shape[2]:
                meta.channel_count = image_shape[2]
        h = image_shape[0]
        w = image_shape[1]
        if (meta.im_size[0] !=w)|(meta.im_size[1]!=h):
            meta.set_new_im_size([w,h])
            meta_changed = True
        return meta_changed, meta
    def get_det_size_by_options(self,det_style = 0):
        if det_style == 0:
            det_size = 13
        elif det_style == self.DET_STYLE_CBOX:
            det_size = 13
        elif det_style == self.DET_STYLE_NEURO_DET:
            det_size = 13
        elif det_style == 12:
            det_size = 8
        return det_size
    def encode_detections(self,detections: [], det_style = 0, input_box_style = None):
        n_det = len(detections)
        det_header_b_size = 4
        det_size = self.get_det_size_by_options(det_style)
        # if det_style == 0:
        #     det_size = 13
        # elif det_style == self.DET_STYLE_CBOX:
        #     det_size = 13
        # elif det_style == self.DET_STYLE_NEURO_DET:
        #     det_size = 13
        # elif det_style == 12:
        #     det_size = 8
        # print(f'encode det size: {det_size}')
        detections_pack = bytearray(det_header_b_size+det_size*n_det)
        detections_pack[0:2] = int(n_det).to_bytes(2, self.b_order, signed=False)
        detections_pack[2:3] = int(det_style).to_bytes(1, self.b_order, signed=False)
        detections_pack[3:4] = int(det_size).to_bytes(1, self.b_order, signed=False)
        offset = det_header_b_size
        if det_style == 0:
            for det in detections:
                # det.print()
                detections_pack[offset:offset+2] = det.id.to_bytes(2, self.b_order, signed=False)
                detections_pack[offset+2:offset + 3] = det.obj_class.to_bytes(1, self.b_order, signed=False)
                detections_pack[offset + 3:offset + 5] = int(10000*det.p).to_bytes(2, self.b_order, signed=False)
                detections_pack[offset + 5:offset + 7] = int(det.bbox[0]).to_bytes(2, self.b_order, signed=False)
                detections_pack[offset + 7:offset + 9] = int(det.bbox[1]).to_bytes(2, self.b_order, signed=False)
                detections_pack[offset + 9:offset + 11] = int(det.bbox[2]).to_bytes(2, self.b_order, signed=False)
                detections_pack[offset + 11:offset + 13] = int(det.bbox[3]).to_bytes(2, self.b_order, signed=False)
                offset+=det_size

        elif det_style == self.DET_STYLE_NEURO_DET:
            for det in detections:
                # det.print()
                detections_pack[offset:offset+2] = det.id.to_bytes(2, self.b_order, signed=False)
                detections_pack[offset+2:offset + 3] = det.obj_class.to_bytes(1, self.b_order, signed=False)
                detections_pack[offset + 3:offset + 5] = int(10000*det.p).to_bytes(2, self.b_order, signed=False)
                detections_pack[offset + 5:offset + 7] = int(det.centered_box[0]).to_bytes(2, self.b_order, signed=False)
                detections_pack[offset + 7:offset + 9] = int(det.centered_box[1]).to_bytes(2, self.b_order, signed=False)
                detections_pack[offset + 9:offset + 11] = int(det.centered_box[2]).to_bytes(2, self.b_order, signed=False)
                detections_pack[offset + 11:offset + 13] = int(det.centered_box[3]).to_bytes(2, self.b_order, signed=False)
                offset+=det_size
        elif det_style == 12:
            for det in detections:
                # det.print()
                if input_box_style == self.DET_STYLE_CVT:
                    box = cv.boundingRect(det)
                else:
                    box = det.bbox
                detections_pack[offset:offset+2] = int(box[0]).to_bytes(2, self.b_order, signed=False)
                detections_pack[offset+2:offset + 4] = int(box[1]).to_bytes(2, self.b_order, signed=False)
                detections_pack[offset+4:offset + 6] = int(box[2]).to_bytes(2, self.b_order, signed=False)
                detections_pack[offset+6:offset + 8] = int(box[3]).to_bytes(2, self.b_order, signed=False)
                offset+=det_size
                # print(offset)
        return detections_pack

    def encode_dets_header(self,n_dets,det_style = 0):
        '''
        Кодирование параметров разметки детектирований: кол-во детектирований, стиль разметки, размер записи
        '''
        det_header_b_size = 4
        dets_header = bytearray(det_header_b_size)
        det_size = self.get_det_size_by_options(det_style)
        # print(f'encode det size: {det_size}')
        # detections_pack = bytearray(det_header_b_size + det_size * n_dets)
        dets_header[0:2] = int(n_dets).to_bytes(2, self.b_order, signed=False)
        dets_header[2:3] = int(det_style).to_bytes(1, self.b_order, signed=False)
        dets_header[3:4] = int(det_size).to_bytes(1, self.b_order, signed=False)
        return dets_header

    def decode_dets_header(self,buffer,offset = 0):
        det_header_b_size = 4
        n_det = int.from_bytes(buffer[offset:offset+2], self.b_order, signed=False)
        det_style = int.from_bytes(buffer[offset+2:offset+3], self.b_order, signed=False)
        det_size = int.from_bytes(buffer[offset+3:offset+4], self.b_order, signed=False)
        offset+=det_header_b_size
        return n_det,det_style,offset

    def encode_single_image(self,image,quality):
        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), quality]
        result, imgencode = cv.imencode(".jpg", image, encode_param)
        img_in_bytes = imgencode.tobytes()
        size_description = len(img_in_bytes).to_bytes(3, self.b_order, signed=False)
        return size_description+img_in_bytes

    def decode_single_image(self,buffer,offset):
        img_b_size = int.from_bytes(buffer[offset:offset + 3], self.b_order, signed=False)
        offset += 3
        img_from_bytes = cv.imdecode(np.frombuffer(buffer[offset:offset + img_b_size],
                                                   dtype=np.uint8), cv.IMWRITE_JPEG_QUALITY)

        offset+=img_b_size
        return img_from_bytes,offset



    def encode_single_detection(self,det, det_style = 0, input_box_style = None):
        det_size = self.get_det_size_by_options(det_style)
        # print(f'encode det size: {det_size}')
        detection_bytes = bytearray(det_size)
        offset = 0
        if det_style == 0:
            detection_bytes[offset:offset+2] = det.id.to_bytes(2, self.b_order, signed=False)
            detection_bytes[offset+2:offset + 3] = det.obj_class.to_bytes(1, self.b_order, signed=False)
            detection_bytes[offset + 3:offset + 5] = int(10000*det.p).to_bytes(2, self.b_order, signed=False)
            detection_bytes[offset + 5:offset + 7] = int(det.bbox[0]).to_bytes(2, self.b_order, signed=False)
            detection_bytes[offset + 7:offset + 9] = int(det.bbox[1]).to_bytes(2, self.b_order, signed=False)
            detection_bytes[offset + 9:offset + 11] = int(det.bbox[2]).to_bytes(2, self.b_order, signed=False)
            detection_bytes[offset + 11:offset + 13] = int(det.bbox[3]).to_bytes(2, self.b_order, signed=False)
            offset+=det_size

        elif det_style == self.DET_STYLE_NEURO_DET:
            detection_bytes[offset:offset+2] = det.id.to_bytes(2, self.b_order, signed=False)
            detection_bytes[offset+2:offset + 3] = det.obj_class.to_bytes(1, self.b_order, signed=False)
            detection_bytes[offset + 3:offset + 5] = int(10000*det.p).to_bytes(2, self.b_order, signed=False)
            detection_bytes[offset + 5:offset + 7] = int(det.centered_box[0]).to_bytes(2, self.b_order, signed=False)
            detection_bytes[offset + 7:offset + 9] = int(det.centered_box[1]).to_bytes(2, self.b_order, signed=False)
            detection_bytes[offset + 9:offset + 11] = int(det.centered_box[2]).to_bytes(2, self.b_order, signed=False)
            detection_bytes[offset + 11:offset + 13] = int(det.centered_box[3]).to_bytes(2, self.b_order, signed=False)
            offset+=det_size

        elif det_style == 12:
            if input_box_style == self.DET_STYLE_CVT:
                box = cv.boundingRect(det)
            else:
                box = det.bbox
            detection_bytes[offset:offset+2] = int(box[0]).to_bytes(2, self.b_order, signed=False)
            detection_bytes[offset+2:offset + 4] = int(box[1]).to_bytes(2, self.b_order, signed=False)
            detection_bytes[offset+4:offset + 6] = int(box[2]).to_bytes(2, self.b_order, signed=False)
            detection_bytes[offset+6:offset + 8] = int(box[3]).to_bytes(2, self.b_order, signed=False)
            offset+=det_size
            # print(offset)
        return detection_bytes,det_size

    def decode_single_detection(self,buffer,offset,det_style):
        det_size = self.get_det_size_by_options(det_style)
        if det_style ==0:
            det = Detection()
            det.id = int.from_bytes(buffer[offset:offset+2], self.b_order, signed=False)
            det.obj_class = int.from_bytes(buffer[offset+2:offset + 3], self.b_order, signed=False)
            det.p = int.from_bytes(buffer[offset + 3:offset + 5], self.b_order, signed=False)/10000
            det.bbox[0] = int.from_bytes(buffer[offset + 5:offset + 7], self.b_order, signed=False)
            det.bbox[1] = int.from_bytes(buffer[offset + 7:offset + 9], self.b_order, signed=False)
            det.bbox[2] = int.from_bytes(buffer[offset + 9:offset + 11], self.b_order, signed=False)
            det.bbox[3] = int.from_bytes(buffer[offset + 11:offset + 13], self.b_order, signed=False)
            offset+=det_size
        elif det_style == self.DET_STYLE_NEURO_DET:
            det = Detection_centered()
            det.id = int.from_bytes(buffer[offset:offset+2], self.b_order, signed=False)
            det.obj_class = int.from_bytes(buffer[offset+2:offset + 3], self.b_order, signed=False)
            det.p = int.from_bytes(buffer[offset + 3:offset + 5], self.b_order, signed=False)/10000
            det.centered_box[0] = int.from_bytes(buffer[offset + 5:offset + 7], self.b_order, signed=False)
            det.centered_box[1] = int.from_bytes(buffer[offset + 7:offset + 9], self.b_order, signed=False)
            det.centered_box[2] = int.from_bytes(buffer[offset + 9:offset + 11], self.b_order, signed=False)
            det.centered_box[3] = int.from_bytes(buffer[offset + 11:offset + 13], self.b_order, signed=False)
            offset+=det_size
        elif det_style == 12:
            new_bbox = [0,0,0,0]
            new_bbox[0] = int.from_bytes(buffer[offset + 0:offset + 2], self.b_order, signed=False)
            new_bbox[1] = int.from_bytes(buffer[offset + 2:offset + 4], self.b_order, signed=False)
            new_bbox[2] = int.from_bytes(buffer[offset + 4:offset + 6], self.b_order, signed=False)
            new_bbox[3] = int.from_bytes(buffer[offset + 6:offset + 8], self.b_order, signed=False)
            offset += det_size
            det = new_bbox
        else:
            det = None
        return det,offset

    def decode_detections(self,buffer, offset = 0):
        det_header_b_size = 4
        n_det = int.from_bytes(buffer[offset:offset+2], self.b_order, signed=False)
        det_style = int.from_bytes(buffer[offset+2:offset+3], self.b_order, signed=False)
        det_size = int.from_bytes(buffer[offset+3:offset+4], self.b_order, signed=False)
        detections = []
        mm = []
        offset+=det_header_b_size
        if det_style ==0:
            # print('from decode : n')
            # print(n_det)
            for i in range(n_det):
                det = Detection()
                det.id = int.from_bytes(buffer[offset:offset+2], self.b_order, signed=False)
                det.obj_class = int.from_bytes(buffer[offset+2:offset + 3], self.b_order, signed=False)
                det.p = int.from_bytes(buffer[offset + 3:offset + 5], self.b_order, signed=False)/10000
                det.bbox[0] = int.from_bytes(buffer[offset + 5:offset + 7], self.b_order, signed=False)
                det.bbox[1] = int.from_bytes(buffer[offset + 7:offset + 9], self.b_order, signed=False)
                det.bbox[2] = int.from_bytes(buffer[offset + 9:offset + 11], self.b_order, signed=False)
                det.bbox[3] = int.from_bytes(buffer[offset + 11:offset + 13], self.b_order, signed=False)
                detections.append(copy.deepcopy(det))
                offset+=det_size
        elif det_style == self.DET_STYLE_NEURO_DET:
            for i in range(n_det):
                det = Detection_centered()
                det.id = int.from_bytes(buffer[offset:offset+2], self.b_order, signed=False)
                det.obj_class = int.from_bytes(buffer[offset+2:offset + 3], self.b_order, signed=False)
                det.p = int.from_bytes(buffer[offset + 3:offset + 5], self.b_order, signed=False)/10000
                det.centered_box[0] = int.from_bytes(buffer[offset + 5:offset + 7], self.b_order, signed=False)
                det.centered_box[1] = int.from_bytes(buffer[offset + 7:offset + 9], self.b_order, signed=False)
                det.centered_box[2] = int.from_bytes(buffer[offset + 9:offset + 11], self.b_order, signed=False)
                det.centered_box[3] = int.from_bytes(buffer[offset + 11:offset + 13], self.b_order, signed=False)
                detections.append(copy.deepcopy(det))
                offset+=det_size
        elif det_style == 12:
            for i in range(n_det):
                new_bbox = [0,0,0,0]
                new_bbox[0] = int.from_bytes(buffer[offset + 0:offset + 2], self.b_order, signed=False)
                new_bbox[1] = int.from_bytes(buffer[offset + 2:offset + 4], self.b_order, signed=False)
                new_bbox[2] = int.from_bytes(buffer[offset + 4:offset + 6], self.b_order, signed=False)
                new_bbox[3] = int.from_bytes(buffer[offset + 6:offset + 8], self.b_order, signed=False)
                offset += det_size
                detections.append(copy.deepcopy(new_bbox))


        return detections, offset
if __name__ == '__main__':
    contructor = V_constructor()
    #
    # im = cv.imread(r'0.jpeg', 0)
    '''
    detection = Detection([870,600,950,650],1,0.8,0)
    detection2 = Detection([970, 700, 1050, 750], 1, 0.8, 1)
    detections = []
    detections.append(detection)
    detections.append(detection2)
    # # for i in range(10):
    # #     detections.append(Detection([870,600,950,650],1,0.8,0))
    # # print(detections)
    # # for det in detections:
    # #     print(det.bbox)
    # #
    # # print(detections[4].p)
    #
    #
    meta_im = Image_meta(10,20,[4000,4000],[9,9])
    meta_im.print()

    bytes = contructor.encode_frame_description(meta_im)
    recovered_meta,offset = contructor.decode_frame_description(bytes,0)
    recovered_meta.print()



    by_d = contructor.encode_detections(detections,12)

    decoded, offset = contructor.decode_detections(by_d,0)
    print(decoded)


    mes_131 = contructor.build_message_131(detections,meta_im,0)

    header, offset = contructor.parse_header(mes_131,0)
    print(header)
    header.print()
    # meta_r,offset = contructor.decode_frame_description(mes_131,offset)
    meta_r,dets_r,offset = contructor.parse_message_131(mes_131,offset)

    meta_r.print()
    print(dets_r)


    print('<<Проверка запроса на распознавание>>')
    targets = [[0,68.4,15.5],[1,44.1,20.0]]
    stamp = time.time()
    mes_56 = contructor.build_t_recognition_req(stamp,targets)
    header, offset = contructor.parse_header(mes_56, 0)

    header.print()
    timestamp,targetings,offset = contructor.parse_t_recognition_req(mes_56, offset)

    print(f'original stamp: {stamp}, parsed: {timestamp}')
    print(targets)
    print(targetings)

    # msg_3 = contructor.build_message_3(im,detections, meta_im)
    #
    # f = open('test.bin','wb')
    # f.write(msg_3)
    # f.close()
    #
    # f_r = open('test.bin','rb')
    # buffer = f_r.read()
    # f_r.close()
    # #
    # head, offset = contructor.parse_header(buffer)
    # if head.message_type == 3:
    #     print('message 3 parsing')
    #     meta_0, offset0 = contructor.decode_frame_description(buffer,offset)
    #     print('!!!!!!!')
    #     meta_0.print()
    #     meta, image, detections, offset = contructor.parse_message_3(buffer, offset)
    #     meta.print()
    #     print('Кол-во обнаружений',len(detections))
    #     for d in detections:
    #         draw_detection(image,d)
    #     cv.imshow('parsed', cv.resize(image,(1000,1000)))
    #     cv.waitKey()
    #
    # roi = [20,30,50,60]
    # roi2 = [25,35,55,65]
    # bbs = contructor.encode_roi(roi)
    # print(bbs)
    #
    # returned, offset = contructor.decode_roi(bbs,0)
    # print(returned)
    #
    # meta = Image_meta(10,20,[1000,1000],[9,9])
    # buf = contructor.build_roi_req(meta,[roi],55)
    #
    # meta, areas, offset = contructor.parse_roi_req(buf,contructor.header_0.size)
    # meta.print()
    # print(areas)

    print('Проверка msg2 с центрованными детектированиями нейросети')
    det1 = Detection_centered()
    det1.centered_box = [906,629,52,44]
    print(det1.to_string())
    dets = [det1]
    contructor.detection_style = contructor.DET_STYLE_NEURO_DET
    buf = contructor.build_message_2(dets,meta_im)
    print(buf)
    header, offset = contructor.parse_header(buf,0)
    header.print()
    meta,dets,offset = contructor.parse_message_2(buf,offset)
    meta.print()
    print(dets[0].to_string())

    print('Проверка кодирования/декодирования изображения')
    img = cv.resize(cv.imread('test_2.jpg'),(50,50))
    # cv.imshow('before encode',img)
    # cv.waitKey()
    encoded_img = contructor.encode_single_image(img,100)
    print(f'img size in bytes: {len(encoded_img)}')
    decoded_img,offset = contructor.decode_single_image(encoded_img,0)
    # cv.imshow('decoded',cv.resize(decoded_img,(300,300)))
    # cv.waitKey()
    # contructor.encode_single_image()

    print('Проверка кодирования/декодирования Пакета 5')
    # buf, size = contructor.encode_single_detection(det1,contructor.detection_style)
    # print(buf)
    # det,offset = contructor.decode_single_detection(buf,0,contructor.detection_style)
    # print(det.to_string())
    buf = contructor.build_message_5([det1],[img],meta_im)
    header, offset = contructor.parse_header(buf, 0)
    header.print()
    meta, dets, images, offset = contructor.parse_message_5(buf, offset)
    meta.print()
    print(f'dets count = {len(dets)}')
    # cv.imshow('got',images[0])
    # cv.waitKey()
    # # print(dets[0].to_string())

    copt_full_image = cv.imread('0.jpeg', 0)
    image_copy = copy.deepcopy(copt_full_image)
    # x1,y1 = det1.left_top()
    # x2,y2 = det1.right_bottom()
    det1 = Detection_centered()
    det1.centered_box = [906, 629, 52, 44]
    det1.draw(image_copy)
    det2 = Detection_centered()
    det2.centered_box = [916, 639, 52, 44]
    dets = [det1, det2]
    det2.draw(image_copy)
    cv.imshow('full',image_copy)
    cv.waitKey()
    # cv.imshow('ww',copt_full_image[y1: y2, x1: x2])
    # cv.waitKey()


    print('Проверка кодирования/декодирования Пакета 5 с автоматической нарезкой изображений')
    buf = contructor.build_message_5(dets,copt_full_image,meta_im)
    header, offset = contructor.parse_header(buf, 0)
    header.print()
    meta, dets_dec, images, offset = contructor.parse_message_5(buf, offset)
    meta.print()
    print(f'dets count = {len(dets_dec)}')
    for i, d in enumerate(dets_dec):
        print(d.to_string())
        cv.imshow(f'{i}',images[i])
    cv.waitKey()
    '''
    print('Проверка сообщений для настройки камер')
    mes = contructor.build_message_61([False,True],[67433.9,45.9,125])
    header,offset = contructor.parse_header(mes)
    header.print()
    if header.message_type == 61:
        flags, vals, offset = contructor.parse_message_61(mes,offset)
        print(f'flags {flags}')
        print(f'vals {vals}')

    print('Проверка сообщений для настройки обработчиков')
    mes = contructor.build_message_62([10, 3, 16])
    header, offset = contructor.parse_header(mes)
    header.print()
    if header.message_type == 62:
        vals, offset = contructor.parse_message_62(mes, offset)
        print(f'vals {vals}')

    print('Проверка сообщений для поворотов ОПУ')
    mes = contructor.build_message_71(1234.4567,234.97,34.23)
    header, offset = contructor.parse_header(mes)
    header.print()
    if header.message_type == 71:
        stamp,dir,reserved,offset = contructor.parse_message_71(mes, offset)
        print(f'stamp {stamp}\ndirection {dir}\nreserved {reserved}\n{offset}')
        print(round(dir[0],2))
        print(round(dir[1],2))

    print('Проверка сообщений для изменения режимов работы сервера')
    mes = contructor.build_message_60(0.0,1,[-35,199,3])
    header, offset = contructor.parse_header(mes)
    header.print()
    if header.message_type == 60:
        stamp,mode,reserved,offset = contructor.parse_message_60(mes, offset)
        print(f'stamp {stamp}\nmode {mode}\nreserved {reserved}\n{offset}')


