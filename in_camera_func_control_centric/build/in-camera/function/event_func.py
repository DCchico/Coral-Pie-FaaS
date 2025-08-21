import logging
import numpy as np
import cv2
import time
import json
import collections

from math import floor, ceil
from sort.sort import *

from adaptive_hist import adaptive_hist

SLogger = logging.getLogger('RPi2')


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class FPS:
    def __init__(self, avarageof=10):
        self.frametimestamps = collections.deque(maxlen=avarageof)

    def __call__(self):
        self.frametimestamps.append(time.time())
        if(len(self.frametimestamps) > 1):
            return len(self.frametimestamps)/(self.frametimestamps[-1]-self.frametimestamps[0])
        else:
            return 0.0


def timing(f):
    def _decorator(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        SLogger.debug("%s function took %.3f ms" % (f.__name__, (time2 - time1) * 1000.0))
        return ret
    return _decorator


def classtiming(f):
    def _decorator(self, *args, **kwargs):
        time1 = time.time()
        ret = f(self, *args, **kwargs)
        time2 = time.time()
        SLogger.debug("%s's %s function took %.3f ms" % (type(self).__name__, f.__name__, (time2 - time1) * 1000.0))
        return ret
    return _decorator


@timing
def parse_load(socket):
    obj = socket.recv_pyobj()
    if obj is None:
        return None

    try:
        rawimage, bboxes = obj
        nparr = np.fromstring(rawimage, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        SLogger.error('Unable to parse the recevied object. Error: %s' % e)
    else:
        return (rawimage, image, bboxes)

@timing
def load_opencv_PIL(pil_image):
    opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return opencvImage

@timing
def frame_storage(vstore, camera_name, frame_id, frame, bboxes):
    vstore.push_frame(camera_name, frame_id, frame, bboxes.tolist())


@timing
def vertex_storage(tgraph, camera_name, tracklet):
    vehid = tracklet.id
    first_frame = tracklet.start_update
    last_frame = tracklet.last_update
    vertexid = tgraph.addDetection('veh%d' % vehid, camera_name, time.time(), '%d-%d' % (first_frame, last_frame))
    return vertexid

@timing
def edge_storage(tgraph, matched_vertices, target_vertex):
    for (vertexid, confidence) in matched_vertices:
        tgraph.linkDetection(vertexid, target_vertex, confidence)


@timing
def feature_extraction_adaptive_histogram(tracklet):
    # pick the median one
    mid = int(len(tracklet.tracklet) / 2)
    bbox = tracklet.tracklet[mid]
    hist = adaptive_hist(bbox.bbox_image)
    return hist


@timing
def messaging(pubsub, vertexid, vehid, hist):
    event = {'vertexid': vertexid,
             'vehid': vehid,
             'camera': pubsub.cname,
             'timestamp': time.time(),
             'hist': hist}
    json_event = json.dumps(event, cls=NumpyEncoder)
    pubsub.publishMessage(json_event)


def listener_func(pubsub, pool):
    while True:
        topic, message_data = pubsub.receiveData()
        event = json.loads(message_data)
        event["hist"] = np.asarray(event["hist"])
        pool.push(event)
        SLogger.debug('Recevied event %s-%s' % (topic, message_data))

class BoundingBox:

    def __init__(self, frameid, frame, bbox):
        self.frameid = frameid
        self.bbox_image = frame[floor(bbox[1]):ceil(bbox[3]), floor(bbox[0]):ceil(bbox[2])]
        self.bbox = bbox
        # self.frame = frame.copy()


class Tracklet:
    """
    A track object for every vehicle before it leaves the camera
    """

    def __init__(self, id):
        self.id = id
        self.vertexid = None
        self.tracklet = []

    def update(self, frameid, frame, bbox):
        if len(self.tracklet) == 0:
            self.start_update = frameid
        self.last_update = frameid
        self.tracklet.append(BoundingBox(frameid, frame, bbox))

class VehicleTracking:

    def __init__(self, max_age=100, min_hits=1):
        self.max_age = max_age
        self.mot_tracker = Sort(max_age, min_hits)
        self.tracklets = {}

    @classtiming
    def sort_update(self, bboxs):
        self.trackers = self.mot_tracker.update(np.asarray(bboxs))
        return self.trackers

    @classtiming
    def status_update(self, tgraph, cameraid, frameid, frame):
        entering_vehicles = []
        for t in self.trackers:
            vid = int(t[4])
            if vid not in self.tracklets:
                self.tracklets[vid] = Tracklet(vid)
                self.tracklets[vid].update(frameid, frame, t[0:4])
                entering_vehicles.append(self.tracklets[vid])
                self.tracklets[vid].vertexid = vertex_storage(tgraph, cameraid, self.tracklets[vid])
            else:
                self.tracklets[vid].update(frameid, frame, t[0:4])

        leaving_vehicle = []
        for tlet in self.tracklets.values():
            print("tlet.last_update ======== %d", tlet.last_update)
            if frameid - tlet.last_update > self.max_age:
                leaving_vehicle.append(tlet)

        for tlet in leaving_vehicle:
            del self.tracklets[tlet.id]
        ret = {"entering_vehicles": entering_vehicles,
               "leaving_vehicle": leaving_vehicle}
        return ret
