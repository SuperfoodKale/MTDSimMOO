import pandas as pd

class CostMetricStatistics:
    def __init__(self):
        self._records = []

    def append(self, timestamp, network):
        total_downtime = sum(h.get_downtime() for h in network.get_host_objects())
        #total_latency = sum(h.get_latency() for h in network.get_host_objects())
        total_agent_time = sum(h.get_agent_time() for h in network.get_host_objects())
        self._records.append({
            "time": timestamp,
            "downtime": total_downtime,
            #"latency": total_latency,
            "agent_time": total_agent_time
        })
    def get_record(self):
        return pd.DataFrame(self._records)