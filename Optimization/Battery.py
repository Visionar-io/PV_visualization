class Battery():
    def __init__(self,max_capacity, minimum_discharge, time_span):
        self.max_capacity = max_capacity
        self.minimum_discharge = minimum_discharge
        self.time_span = time_span
        self.capacity = 0
        self.capacity_hist = []
    
    def charge(self, kw):
        if self.capacity + kw*self.time_span > self.max_capacity:
            pass
        else:
            self.capacity += kw*self.time_span

    def discharge(self, kw):
        if self.capacity - kw*self.time_span < self.max_capacity*self.minimum_discharge:
            pass
        else:
            self.capacity -= kw*self.time_span

Bateria = Battery(max_capacity=100,
                  minimum_discharge=.2,
                  time_span=5)
print(Bateria.capacity)
Bateria.charge(20)
print(Bateria.capacity)
Bateria.discharge(20)
print(Bateria.capacity)
