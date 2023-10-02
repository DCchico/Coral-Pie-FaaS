import datetime
import logging
import sys
import os
import time
from gremlin_python import statics
from gremlin_python.driver.driver_remote_connection import \
    DriverRemoteConnection
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.process.graph_traversal import __
from gremlin_python.process.strategies import *
from gremlin_python.process.traversal import (Barrier, Bindings, Cardinality,
                                              Column, Direction, Operator,
                                              Order, P, Pop, Scope, T,
                                              WithOptions)
from gremlin_python.structure.graph import Graph


class TrajectoryGraph:
    # use for binding
    LABEL = "label"
    CAMID = "camId"
    TIME = "time"
    IMAGE = "image"
    OUT_V = "outV"
    IN_V = "inV"
    LIMIT = "limit"
    VID = "vid"
    FEA = "feature"
    CONFIDENCE = "confidence"
    INDEX = "index"

    def __init__( self ):
        self.b = Bindings()
        self.graph = Graph()
        self.connection = DriverRemoteConnection('ws://10.100.1.239:8182/gremlin','g')
        self.g = self.graph.traversal().withRemote(self.connection)
        logging.info("Connected")
    
    def addDetection(self, vehId, camId, timestamp, index):
        v = self.g.addV(self.b.of(TrajectoryGraph.LABEL, vehId))\
            .property(TrajectoryGraph.CAMID, self.b.of(TrajectoryGraph.CAMID, camId))\
            .property(TrajectoryGraph.TIME, self.b.of(TrajectoryGraph.TIME, timestamp))\
            .property(TrajectoryGraph.INDEX, self.b.of(TrajectoryGraph.INDEX, index))\
            .next()
        
        logging.info("Trajectory Vertex v[{}] ({}, {}, {}) created.".format(v, vehId, camId, timestamp))
        
        return v

        
    def linkDetection(self, src, dest, confidence):
        logging.info("Link vertex v[{}] to v[{}]. Confidence {}".format(src, dest, confidence))
        self.g.V(self.b.of(TrajectoryGraph.OUT_V, src))\
            .as_("a")\
            .V(self.b.of(TrajectoryGraph.IN_V, dest))\
            .addE(self.b.of(TrajectoryGraph.LABEL, "next"))\
            .from_("a")\
            .property(TrajectoryGraph.CONFIDENCE, self.b.of(TrajectoryGraph.CONFIDENCE, confidence))\
            .iterate()
    
    def getValueMapById(self,id):
        value = self.g.V(self.b.of(TrajectoryGraph.VID, id)).valueMap(True).next()
        logging.info("Get detection valuemap {} for V[{}]".format(value.keys(), id))
        return value


    def getLatestDetectionsByCamId(self, camId, limit):
        vehIds = self.g.V().has(TrajectoryGraph.CAMID, camId).order().by(TrajectoryGraph.TIME, Order.desc).limit(limit).toList()
        logging.info("LatestDetections by camera {}: {}".format(camId, vehIds))
        return vehIds


    def getNextDetectionsById(self, id, limit):
        #  This can be used to return self
        vehIds = self.g.V(self.b.of(TrajectoryGraph.OUT_V, id)).emit().repeat(__.out()).times(self.b.of(TrajectoryGraph.LIMIT, limit)).toList()
        logging.info("NextDetections from V[{}]: {}".format(id, vehIds))
        return vehIds

    def getPrevDetectionsById(self, id, limit):
        vehIds = self.g.V(self.b.of(TrajectoryGraph.IN_V, id)).repeat(__.in_()).times(limit).emit().toList()
        vehIds = vehIds[::-1]
        logging.info("PrevDetections from V[{}]: {}".format(id, vehIds))
        return vehIds

    def clear(self):
        logging.info("TrajectoryGraph dropped")
        self.g.V().drop().iterate()
    
    def shutdown(self):
        logging.info("TrajectoryGraph closed")
        self.connection.close()
        self.g = None
        self.graph = None
        
    

if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    testGraph = TrajectoryGraph()
    testGraph.clear()

    v1 = testGraph.addDetection("veh1", "cam1", 2000.0, "")
    v2 = testGraph.addDetection("veh1", "cam2", 2000.0, "")
    v3 = testGraph.addDetection("veh1", "cam1", 3000.0, "")

    testGraph.linkDetection(v1, v2, "")
    testGraph.linkDetection(v2, v3, "")


    testGraph.getLatestDetectionsByCamId("cam1", 5);
    testGraph.getNextDetectionsById(v1, 5)
    testGraph.getPrevDetectionsById(v3, 5)
    testGraph.getValueMapById(v1)
    testGraph.getValueMapById(v2)

    testGraph.clear()
    testGraph.shutdown()
















#Comments

        # graph = Graph()
        # g = graph.traversal().withRemote(DriverRemoteConnection('ws://localhost:8182/gremlin','g'))
        # print(g.V())
        # graph = Graph()
        # connection = DriverRemoteConnection('ws://localhost:8182/gremlin', 'g')
        # g = graph.traversal().withRemote(connection)
        # g.addV("hi").id().next()
        # g.addV("hii").id().next()
        # connection = DriverRemoteConnection('ws://127.0.0.1:8182/gremlin', 'g')
        # // The connection should be closed on shut down to close open connections with connection.close()
        # g = graph.traversal().withRemote(connection)
        # g.addV("hello")
        # // Reuse 'g' across the application
        # print(g.V().count().toList())


    # public TrajectoryGraph() {
    #     b = Bindings.instance();
    #     graph = EmptyGraph.instance();
    #     try {
    #         g = graph.traversal().withRemote(CONFIG_FILE);
    #     } catch (Exception e) {
    #         LOGGER.error(e.getMessage(), e);
    #         System.exit(-1);
    #     }
    # }
