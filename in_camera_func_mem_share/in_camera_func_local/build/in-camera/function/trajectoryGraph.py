'''
import datetime
import logging
import sys
import os
import time
import uuid  # For generating IDs

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

    def __init__(self):
        # In-memory storage instead of Gremlin
        self.vertices = {}  # Dictionary to store vertices
        self.edges = {}     # Dictionary to store edges
        logging.info("Connected to in-memory graph")
    
    def addDetection(self, vehId, camId, timestamp, index):
        # Generate a unique ID for the vertex
        vertex_id = str(uuid.uuid4())
        
        # Create vertex with properties
        vertex = {
            "id": vertex_id,
            TrajectoryGraph.LABEL: vehId,
            TrajectoryGraph.CAMID: camId,
            TrajectoryGraph.TIME: timestamp,
            TrajectoryGraph.INDEX: index
        }
        
        # Store the vertex
        self.vertices[vertex_id] = vertex
        
        logging.info("Trajectory Vertex v[{}] ({}, {}, {}) created.".format(vertex_id, vehId, camId, timestamp))
        
        return vertex_id

    def linkDetection(self, src, dest, confidence):
        if src not in self.vertices or dest not in self.vertices:
            logging.warning(f"Cannot link: vertex {src} or {dest} not found")
            return
            
        # Generate a unique ID for the edge
        edge_id = f"{src}_{dest}"
        
        # Create edge with properties
        edge = {
            "id": edge_id,
            "from": src,
            "to": dest,
            TrajectoryGraph.LABEL: "next",
            TrajectoryGraph.CONFIDENCE: confidence
        }
        
        # Store the edge
        self.edges[edge_id] = edge
        
        logging.info("Link vertex v[{}] to v[{}]. Confidence {}".format(src, dest, confidence))
    
    def getValueMapById(self, id):
        if id not in self.vertices:
            logging.warning(f"Vertex {id} not found")
            return {}
        
        value = self.vertices[id].copy()
        logging.info("Get detection valuemap {} for V[{}]".format(value.keys(), id))
        return value

    def getLatestDetectionsByCamId(self, camId, limit):
        # Filter vertices by camId
        filtered_vertices = [v for v in self.vertices.values() if v.get(TrajectoryGraph.CAMID) == camId]
        
        # Sort by timestamp (descending)
        sorted_vertices = sorted(filtered_vertices, key=lambda v: v.get(TrajectoryGraph.TIME, 0), reverse=True)
        
        # Limit results
        limited_vertices = sorted_vertices[:limit]
        
        # Extract IDs
        vehIds = [v["id"] for v in limited_vertices]
        
        logging.info("LatestDetections by camera {}: {}".format(camId, vehIds))
        return vehIds

    def getNextDetectionsById(self, id, limit):
        if id not in self.vertices:
            logging.warning(f"Vertex {id} not found")
            return []
        
        # Simplified implementation: just find edges where this vertex is the source
        result = [id]  # Include the starting vertex in results
        current = id
        count = 0
        
        while count < limit:
            # Find outgoing edges
            outgoing = [e for e in self.edges.values() if e["from"] == current]
            if not outgoing:
                break
                
            # Take the first edge and follow it
            next_id = outgoing[0]["to"]
            result.append(next_id)
            current = next_id
            count += 1
        
        logging.info("NextDetections from V[{}]: {}".format(id, result))
        return result

    def getPrevDetectionsById(self, id, limit):
        if id not in self.vertices:
            logging.warning(f"Vertex {id} not found")
            return []
            
        # Simplified implementation: just find edges where this vertex is the destination
        result = []
        current = id
        count = 0
        
        while count < limit:
            # Find incoming edges
            incoming = [e for e in self.edges.values() if e["to"] == current]
            if not incoming:
                break
                
            # Take the first edge and follow it backwards
            prev_id = incoming[0]["from"]
            result.insert(0, prev_id)  # Insert at beginning to maintain order
            current = prev_id
            count += 1
            
        result.append(id)  # Add the starting vertex at the end
        
        logging.info("PrevDetections from V[{}]: {}".format(id, result))
        return result

    def clear(self):
        self.vertices = {}
        self.edges = {}
        logging.info("TrajectoryGraph dropped (in-memory storage cleared)")
    
    def shutdown(self):
        self.vertices = {}
        self.edges = {}
        logging.info("TrajectoryGraph closed (in-memory storage cleared)")
        
if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    testGraph = TrajectoryGraph()
    testGraph.clear()

    v1 = testGraph.addDetection("veh1", "cam1", 2000.0, "")
    v2 = testGraph.addDetection("veh1", "cam2", 2000.0, "")
    v3 = testGraph.addDetection("veh1", "cam1", 3000.0, "")

    testGraph.linkDetection(v1, v2, "")
    testGraph.linkDetection(v2, v3, "")

    testGraph.getLatestDetectionsByCamId("cam1", 5)
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
'''
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
        self.connection = DriverRemoteConnection('ws://128.110.217.193:8182/gremlin','g')
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