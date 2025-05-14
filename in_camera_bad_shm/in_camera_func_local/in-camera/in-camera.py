from event_func import *
from trajectoryGraph import TrajectoryGraph
from candidatePool import CandidatePool

from coldstart import coldstart
import logging

# Set the logging level to INFO (or the desired level)
logging.basicConfig(level=logging.INFO)

tgraph = TrajectoryGraph()
vt = VehicleTracking()
pool = CandidatePool()
coldstart(tgraph)

camera_id = 0
frame_id = 0
image = cv2.imread("coldstart.jpeg")
bboxes = [[183.3546, 233.2469, 267.5851, 296.2590, 100]]

logging.info("Frame id: %d", frame_id)      

tracked_bboxes = vt.sort_update(bboxes)
logging.info("Updated Vehicle Tracklets with new BBoxes")   

leaving_vehicles = vt.status_update(frame_id, image)
logging.info("Extracted %d Vehicles Leaving with age > 3", len(leaving_vehicles))

for vehicle in leaving_vehicles:
    logging.info("Vehicle: %d is leaving" % vehicle.id)
    hist = feature_extraction_adaptive_histogram(vehicle)
    vertexid = vertex_storage(tgraph, camera_id, vehicle)

logging.info("Finished Iteration")