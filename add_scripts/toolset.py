import cv2 as cv
import multiprocessing as mp
import numpy as np
import time
import geometry_lib as glib
import copy

class Check_state:
    def __init__(self, step_timeout):
        self.current_track_id = -1
        self.el_steps = []
        self.az = 0.0
        self.steps_count = -1
        self.step_timeout = step_timeout
        self.last_step_stamp = 0.0

    def timed_out_and_done(self,stamp = time.time()):
        timed_out = False
        done = False
        if self.last_step_stamp<=(stamp-self.step_timeout):
            timed_out = True
        if timed_out&(self.steps_count >= (len(self.el_steps)-1)):
            done = True
        return timed_out, done

    def next_step(self):
        self.steps_count+=1
        self.last_step_stamp = time.time()

    def set_new(self,id,az,el_steps):
        self.current_track_id = id
        self.el_steps = el_steps
        self.az = az
        self.last_step_stamp = 0.0



class Simple_check:
    def __init__(self, id, stamp, flag = False,obj_class = -1):
        self.flag = flag
        self.id = id
        self.stamp = stamp
        self.obj_class = obj_class

    def print(self):
        print(f'Check: {self.id}, {self.stamp}')

class Check_list:
    def __init__(self, timeout = 10):
        self.ch_list = []
        self.check_timeout = timeout

    def update_by_stamp(self,stamp = None):
        if stamp:
            pass
        else:
            stamp = time.time()
        new_list = []
        for check in self.ch_list:
            if check.stamp >=(stamp-self.check_timeout):
                new_list.append(check)
            else:
                pass
    def put_new_check(self,id,stamp,flag = True, obj_class = -1):
        new_check = Simple_check(id,stamp,flag,obj_class)
        self.ch_list.append(new_check)
    def print(self):
        print('<<Check list>>')
        for ch in self.ch_list:
            ch.print()

    def search_by_id(self, id):
        '''
        Если id есть в списке проверенных, возвращает индекс этого id из списка
        '''
        check_idx = -1
        for i, check in enumerate(self.ch_list):
            if id == check.id:
                check_idx = i
                break
        return check_idx


class Loc_polar_pt:
    def __init__(self,timestamp = 0.0,az=0.0,el = 0.0,r = 0.0,vr= 0.0):
        self.timestamp = timestamp
        self.az = az
        self.el = el
        self.r = r
        self.vr = vr
    def print(self):
        print(f'Locator pt: az = {self.az}, el = {self.el}, r = {self.r}, V_radial = {self.vr}, stamp: {self.timestamp}')

class Locator_track:
    def __init__(self,id):
        self.id = id
        self.dang_flag = 0
        self.obj_type = 0
        self.pts_list = []
    def add_pt(self,pt:Loc_polar_pt):
        self.pts_list.append(pt)
    def get_last_az_el(self):
        if len(self.pts_list)>0:
            last_pt_idx = len(self.pts_list)-1
            return self.pts_list[last_pt_idx].az,self.pts_list[last_pt_idx].el, self.pts_list[last_pt_idx].r
        else:
            return None,None,None
    def get_last_pt(self):
        if len(self.pts_list) > 0:
            last_pt_idx = len(self.pts_list) - 1
            return self.pts_list[last_pt_idx]
        else:
            return None

    def get_prev_pt(self):
        if len(self.pts_list) > 1:
            prev_pt_idx = len(self.pts_list) - 2
            return self.pts_list[prev_pt_idx]
        else:
            return None

    def print(self):
        print(f'<<TRACK {self.id}>>')
        if len(self.pts_list)>0:
            for pt in self.pts_list:
                pt.print()

class Codec_mini:
    def __init__(self):
        self.b_order = 'little'
    def encode_point(self, xy, stamp):
        buf = bytearray(12)
        buf[0:8] = int(stamp * 1000000).to_bytes(8, self.b_order, signed=False)
        buf[8:10] = int(xy[0]).to_bytes(2, self.b_order, signed=True)
        buf[10:12] = int(xy[1]).to_bytes(2, self.b_order, signed=True)
        return buf

    def decode_point(self,packet, offset = 0):
        timestamp = int.from_bytes(packet[offset:offset + 8], self.b_order, signed=False) / 1000000
        x = int.from_bytes(packet[offset + 8:offset + 10], self.b_order, signed=True)
        y = int.from_bytes(packet[offset + 10:offset + 12], self.b_order, signed=True)
        return [x,y], timestamp,offset+12

    def encode_track_header(self,id,wh,n_points):
        buf = bytearray(7)
        buf[0:2] = int(id+1).to_bytes(2, self.b_order, signed=False)
        buf[2:4] = int(wh[0]).to_bytes(2, self.b_order, signed=False)
        buf[4:6] = int(wh[1]).to_bytes(2, self.b_order, signed=False)
        buf[6:7] = int(n_points).to_bytes(1, self.b_order, signed=False)
        return buf

    def decode_track_header(self,packet,offset):
        id = int.from_bytes(packet[offset + 0:offset + 2], self.b_order, signed=False)-1
        w = int.from_bytes(packet[offset + 2:offset + 4], self.b_order, signed=False)
        h = int.from_bytes(packet[offset + 4:offset + 6], self.b_order, signed=False)
        n_points = int.from_bytes(packet[offset + 6:offset + 7], self.b_order, signed=False)
        return id,[w,h],n_points, offset+7

    def decode_track_to_simple_track(self,packet,offset):
        tr_id, size, n_points, offset = self.decode_track_header(packet,offset)
        print(f'tr_id, size, n_points, offset\n{tr_id, size, n_points, offset}')
        new_track = Simple_track(tr_id)
        pt_list = []
        for i in range(n_points):
            x_y, timestamp,offset = self.decode_point(packet,offset)
            new_det = Detection_centered([x_y[0],x_y[1],size[0],size[1]],0,0,tr_id)
            new_det.stamp = timestamp
            pt_list.append(new_det)
            new_track.points_list = pt_list
            # new_track.points_list.append(new_det)
        return new_track, offset

class Detection_centered:
    def __init__(self, centered_box=[0.0, 0.0, 0.0, 0.0], obj_class=0, obj_p=0.9, obj_id=0):
        self.id = obj_id
        self.obj_class = obj_class
        self.p = obj_p
        self.centered_box = centered_box
        self.stamp = 0.0

    def set_center(self,x,y):
        self.centered_box[0] = x
        self.centered_box[1] = y

    def set_wh(self,w,h):
        self.centered_box[2] = w
        self.centered_box[3] = h

    def to_string(self):
        return f'Detection: bbox[{self.centered_box}], class: {self.obj_class}, p: {self.p}, id:{self.id}, stamp: {self.stamp}'
    def draw(self,image,color = (0,0,0),thickness = 6):
        pt1,pt2 = glib.box_cvt_cent2corners_pts(self.centered_box)
        cv.rectangle(image,pt1,pt2,color,thickness )
        cv.putText(image,f'{self.obj_class}[{round(self.p,3)}]',(pt1[0],pt1[1]-10),3,1.3,(0,0,0),2)
    def get_center(self):
        return self.centered_box[0], self.centered_box[1]
    def get_int_center(self):
        return int(self.centered_box[0]), int(self.centered_box[1])
    def get_wh(self):
        return self.centered_box[2],self.centered_box[3]
    # def get_int_center(self):
    #     return int(self.centered_box[0]),int(self.centered_box[1])

    def left_top(self):
        return (int(self.centered_box[0]-self.centered_box[2]/2),int(self.centered_box[1]-self.centered_box[3]/2))

    def right_bottom(self):
        return (int(self.centered_box[0]+self.centered_box[2]/2),int(self.centered_box[1]+self.centered_box[3]/2))

class Simple_track:
    def __init__(self,id = 0,pts_list = []):
        self.id = id
        self.points_list = pts_list

    def to_string(self):
        return f'track id: {self.id}, n points: {len(self.points_list)}'
    def print(self):
        print(self.to_string())
        for d in self.points_list:
            print(d.to_string())
    def draw(self,image,color = (0,0,0),thickness = 20):
        pt1, pt2 = glib.box_cvt_cent2corners_pts(self.points_list[0].centered_box)
        cv.rectangle(image, pt1, pt2, color, thickness)
        cv.putText(image, f'{self.id}', (pt1[0], pt1[1] - 10), 3, 1.3, (0, 0, 0), 2)
        for i in range(len(self.points_list)-1):
            pt1 = self.points_list[i].get_int_center()
            pt2 = self.points_list[i+1].get_int_center()
            cv.line(image,pt1,pt2,color,thickness)
    def draw_annotation(self,image,ann:str,color = (0,0,0),thickness = 2):
        pt1 = self.points_list[0].get_center()
        cv.putText(image, f'{ann}', (pt1[0], pt1[1] - 10), 3, 1.3, (0, 0, 0), 2)

class Simple_buffer:
    '''
    Буфер - сдвиговый регистр
    '''
    def __init__(self,size:int):
        self.size = size
        self.arr = [0]*self.size
        self.current_index = -1
        self.counter = 0
    def put_new(self,value):
        self.current_index+=1
        self.current_index%=self.size
        self.arr[self.current_index] = value
        self.counter+=1
        return self.current_index

    def flush(self):
        for i in range(self.size):
            self.arr[i] = 0
        self.current_index = -1
        self.counter = 0

    def get_sum(self):
        return sum(self.arr)

    def is_full(self):
        if self.counter>=self.size:
            return True
        else:
            return False






class Simple_timed_record:
    '''
    Для хранения записи с меткой времени
    '''
    def __init__(self,value = 0,stamp = 0.0):
        self.timestamp = stamp
        self.value = value
    def set_new(self,value, stamp):
        self.timestamp = stamp
        self.value = value
    def clear(self):
        self.timestamp = 0.0
        self.value = 0
    def add_one(self,stamp):
        self.timestamp = stamp
        self.value+=1
    def get_timed_value(self):
        return self.timestamp, self.value

class Simple_timed_counter:
    def __init__(self,size):
        self.list_size = size
        self.rec_list = []
        for i in range(self.list_size):
            self.rec_list.append(Simple_timed_record())

    def clear_by_latency(self,latency):
        now = time.time()
        for record in self.rec_list:
            if (now- record.timestamp)>=latency:
                record.clear()

    def add_to_idx(self,idx,stamp):
        self.rec_list[idx].add_one(stamp)

    def get_by_idx(self,idx):
        return self.rec_list[idx].get_timed_value()

class Color_selector:
    def __init__(self):

        self.base_color_blue = 0
        self.base_color_green = 1
        self.base_color_red = 2

        self.active_color_id = self.base_color_red

        self.red = (0,0,255)
        self.blue = (255,0,0)
        self.green = (0,255,0)


    def set_active_color(self,color_id):
        self.active_color_id = color_id

    def get_color_by_stamp(self,stamp,start_stamp,stop_stamp):
        k = int(200*max(0,stop_stamp-stamp)/(stop_stamp-start_stamp))
        color = (0,0,0)
        if self.active_color_id ==self.base_color_blue:
            color = (255,k,k)
        elif self.active_color_id ==self.base_color_green:
            color = (k,255,k)
        elif self.active_color_id ==self.base_color_red:
            color = (k,k,255)
        return color



class Simple_targeting:
    def __init__(self):
        self.az = 0
        self.el = 0
        self.h = 0
        self.r_plane = 0
        self.r_3d = 0
        self.stamp = 0.0

    def set_params(self, az,el,h,r_plane,r_3d,stamp):
        self.az = az
        self.el = el
        self.h = h
        self.r_plane = r_plane
        self.r_3d = r_3d
        self.stamp = stamp

    def to_string(self):
        return f'{self.az};{self.el};{self.h};{self.r_plane};{self.r_3d};{self.stamp}'

class Recorder:
    def __init__(self,path):
        self.path = path
        self.file_ = None
        self.line_count = 0

    def start_session(self):
        self.file_ = open(self.path,'w')
    def close_session(self):
        self.file_.close()
        self.file_ = None
        self.line_count = 0
    def write_line(self,text):
        self.file_.write(f'{self.line_count}/ {text} \r')
        self.line_count+=1
class Command:
    def __init__(self,name,val):
        self.name = name
        self.value = val
    def print(self):
        print(f'CMD: {self.name}, Value: {self.value}')


class Mp_dev_interface:
    def __init__(self,size = 5, allow_loose = False):
        self.q_in = mp.Queue(size)
        self.q_out = mp.Queue(size)
        self.allow_loose = allow_loose

    def push_cmd_to_dev(self, cmd):
        if self.allow_loose:
            if self.q_in.full():
                self.q_in.get()
        self.q_in.put(cmd)

    def push_or_loose_cmd_to_dev(self,cmd):
        if not(self.q_in.full()):
            self.q_in.put(cmd)
        else:
            pass

    def push_rep_from_dev(self,cmd):
        if self.allow_loose:
            if self.q_out.full():
                self.q_out.get()
            else:
                pass
        self.q_out.put(cmd)


    def get_rep_from_dev(self):
        got_cmd = False
        if not(self.q_out.empty()):
            if self.allow_loose:
                while not self.q_out.empty():
                    cmd = self.q_out.get()
            else:
                cmd = self.q_out.get()
            got_cmd = True

        else:
            cmd = None
        return got_cmd,cmd

    def get_cmd_to_dev(self):
        got_cmd = False
        if not (self.q_in.empty()):
            if self.allow_loose:
                while not self.q_in.empty():
                    cmd = self.q_in.get()
            else:
                cmd = self.q_in.get()
            got_cmd = True
            # cmd = self.q_in.get()
        else:
            cmd = None
        return got_cmd, cmd
# class Mp_dev_interface:
#     def __init__(self):
#         self.q_in = mp.Queue(5)
#         self.q_out = mp.Queue(5)
#     def push_cmd_to_dev(self, cmd:Command):
#         self.q_in.put(cmd)
#     def push_rep_from_dev(self,cmd:Command):
#         self.q_out.put(cmd)
#
#     def get_rep_from_dev(self):
#         got_cmd = False
#         if not(self.q_out.empty()):
#             got_cmd = True
#             cmd = self.q_out.get()
#         else:
#             cmd = None
#         return got_cmd,cmd
#
#     def get_cmd_to_dev(self):
#         got_cmd = False
#         if not (self.q_in.empty()):
#             got_cmd = True
#             cmd = self.q_in.get()
#         else:
#             cmd = None
#         return got_cmd, cmd


def make_border_mask(shape,pad = 25,style = 'linear'):
    n = pad
    k = 1 / n
    k_array = [0.0] * n
    for i in range(n):
        k_array[i] = i * k
    h_r, w_r = shape
    border_mask_h = np.ones(shape)
    border_mask_v = np.ones(shape)
    for i, k in enumerate(k_array):
        border_mask_v[:, i:i + 1] = k
        border_mask_h[i:i + 1, :] = k
        border_mask_h[h_r - n + i:h_r - n+1 + i, :] = 1-k
        border_mask_v[:, w_r - n + i:w_r - n+1 + i] = 1 - k
    border_mask = cv.multiply(border_mask_v, border_mask_h)
    # test_im = np.ones(shape,np.uint8)
    # test_im*=255
    # cv.multiply(test_im, border_mask, dtype=cv.CV_8U, dst=test_im)
    return border_mask
class Timer:
    def __init__(self, t = time.time()):
        self.start_t = t
        self.stop_t = t

    def start(self):
        self.start_t = time.time()

    def stop(self):
        self.stop_t = time.time()
        return self.stop_t-self.start_t

    def non_stop_elapsed(self):
        return time.time()-self.start_t

if __name__ == "__main__":
    # make_border_mask((700,700))
    # test_recorder = Recorder('test_recort.txt')
    # test_recorder.start_session()
    # test_recorder.write_line('first    dsklaj;')
    # test_recorder.write_line('second 7 272 ')
    # # test_recorder.close_session()
    #
    # counter =Simple_timed_counter(100)
    #
    # for i in range(30):
    #     counter.add_to_idx(0,time.time())
    #     stamp, val = counter.get_by_idx(0)
    #     time.sleep(0.1)
    #     counter.clear_by_latency(0.5)
    #     print(f'{stamp,val}')
    buf = Simple_buffer(5)
    print(buf.arr)
    codec = Codec_mini()
    stamp = 16121491.889988
    xy = [1532,985]
    # print(f'before encoding: {stamp},{xy}')
    buf = codec.encode_point(xy,stamp)
    xy, stamp,offset = codec.decode_point(buf)
    # print(f'after decoding: {stamp},{xy}')
    buf = codec.encode_track_header(2, [30, 26], 3)
    buf += codec.encode_point([120, 130], 0)
    buf += codec.encode_point([121, 131], 1)
    buf += codec.encode_point([122, 132], 2)
    buf += codec.encode_track_header(7,[32,25],3)
    buf +=codec.encode_point([20,30],0)
    buf += codec.encode_point([21, 31], 1)
    buf += codec.encode_point([22, 32], 2)


    tr_id, size, n_points, offset = codec.decode_track_header(buf,0)
    print(f'tr_id {tr_id}, size {size}, n_points {n_points}')
    track,offset = codec.decode_track_to_simple_track(buf,0)
    track.print()
    # tr_id2, size2, n_points2, offset = codec.decode_track_header(buf, offset)
    #
    # print(f'tr_id {tr_id2}, size {size2}, n_points {n_points2}')
    track2, offset2 = codec.decode_track_to_simple_track(buf,offset)
    track2.print()

    print('___________________')
    track.print()
    track2.print()
    # print(track.points_list)
    # print(track2.points_list)
    # track.print()
    # track2.print()
    print('Try not to become mad')
    pack = b'\x83\x00\x1f\x00\x1f\x00\x05\x98L\x08\x81\x07\xf7\x05\x00\xc6\x01\xca\x00\xe6\xe2\x05\x81\x07\xf7\x05\x00a\x02,\x01E\xcc\x02\x81\x07\xf7\x05\x00\x0f\x03S\x01T4\x00\x81\x07\xf7\x05\x00\xc1\x03c\x01\xb7\x16\xfc\x80\x07\xf7\x05\x00\xb3\x04\xfd\x00'
    track, offset = codec.decode_track_to_simple_track(pack, 0)
    track.print()
