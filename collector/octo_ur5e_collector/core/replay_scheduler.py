class ReplayScheduler:
    def __init__(self,events,stale_sec):
        self.events=list(events); self.index=0; self.stale_sec=stale_sec; self.cancelled=False
    def advance(self,progress_sec):
        due=[]
        while self.index<len(self.events) and self.events[self.index]["time_sec"]<=progress_sec:
            due.append(self.events[self.index]); self.index+=1
        return due
    def feedback_stale(self,now,last_feedback): return now-last_feedback>self.stale_sec
    def cancel(self): self.cancelled=True
