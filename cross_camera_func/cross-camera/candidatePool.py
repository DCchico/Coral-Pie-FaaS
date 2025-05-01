import threading
import logging
import time

from adaptive_hist import bhattacharyya as distance

class CandidatePool:
    
    def __init__(self):
        self.pool = []
        self.logger = logging.getLogger("CandidatePool") 
        # self.lock = threading.Lock()
        clean_thread = threading.Thread(target=self.cleanup_loop, args=(10,10))
        clean_thread.start()

    def push(self, event):
        self.pool.append(event)

    def matching(self, target_hist, threshold):
        results = []
        for index in range(len(self.pool)):
            d = distance(target_hist, self.pool[index]["hist"])
            if d < threshold:
                self.pool[index]["matched"] = True
                results.append((self.pool[index]["vertexid"], d))
        return results

    def cleanup_loop(self, delay, time_threshold):
        self.logger.debug("Candidate Pool clean thread launched.")
        while True:
            self.cleanup(time_threshold)
            time.sleep(delay)

    def cleanup(self, time_threshold):
        current_time = time.time()
        for index in reversed(range(len(self.pool))):
            event = self.pool[index]
            if "matched" in event and current_time - event["timestamp"] > time_threshold:
                del self.pool[index]
                self.logger.debug("Event %s evicted from the pool" % event)
        




