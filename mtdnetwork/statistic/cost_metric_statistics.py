import pandas as pd
from mtdnetwork.util import realtime 

class CostMetricStatistics:
    def __init__(self):
        self._records = [{
            "time": 0,
            "elapsed": 0,
            "realtime": 0,
            "downtime": 0,
            "agent_time": 0,
            #"avg_downtime": avg_downtime,
            "downtime_ratio": 0,
            "agent_time_ratio": 0,
            "mtd_actions": 0,
            "mtd_opportunities": 0,
            "mtd_action_ratio": 0,
            
        }]
        self._exp_start_time = realtime.now()
        self._total_agent_time = 0
        self._average_host_downtime = 0
        self._start_time = None
        self._total_mtd_actions = 0
        self._total_mtd_opportunities = 0
        self._mtd_action_ratio = 0.0
        self._total_downtime = 0.0
        


    def append(self, timestamp, network):
        curr_realtime = realtime.now()
        realtime_elapsed = curr_realtime - self._exp_start_time
        #total_downtime = sum(h.get_downtime() for h in network.get_host_objects())
        if self._start_time is None:
            self._start_time = float(timestamp)
        elapsed = max(float(timestamp) - self._start_time, 1.0)

        num_hosts = max(len(network.get_host_objects()), 1)
        #self._avg_host_downtime = total_downtime / num_hosts

        if self._total_mtd_opportunities > 0:
            self._mtd_action_ratio = self._total_mtd_actions / self._total_mtd_opportunities
        else:
            self._mtd_action_ratio = 0

        if realtime_elapsed > 0:
            agent_time_ratio = self._total_agent_time / realtime_elapsed
        else:
            agent_time_ratio = 0

        if realtime_elapsed > 0:
            downtime_ratio = self._total_downtime / realtime_elapsed
        else:
            downtime_ratio = 0
        
        
        self._records.append({
            "time": timestamp,
            "elapsed": elapsed,
            "realtime": realtime_elapsed,
            "downtime": self._total_downtime,
            "agent_time": self._total_agent_time,
            #"avg_downtime": avg_downtime,
            "downtime_ratio": downtime_ratio,
            "agent_time_ratio": agent_time_ratio,
            "mtd_actions": self._total_mtd_actions,
            "mtd_opportunities": self._total_mtd_opportunities,
            "mtd_action_ratio": self._mtd_action_ratio,
            
        })
        
    def get_record(self, timestamp=None, network=None):
        if timestamp is not None and network is not None:
            self.append(timestamp, network)
        #self.append(timestamp, network)
        return pd.DataFrame(self._records)

    def add_agent_time(self, time):
        self._total_agent_time += time

    def add_downtime(self, time):
        self._total_downtime += time

    def add_mtd_executions(self):
        self._total_mtd_actions += 1

    def add_mtd_opportunities(self):
        self._total_mtd_opportunities += 1