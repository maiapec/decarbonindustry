

# from components : variables, inner constraints (like the state of charge and max delivered power, 
#but not the must meet load constraint, that's a system wide constraint) and gas and power input (power input can be negative) + heat output

# component.name
# component._variables
# component._constraints
# component.powerConsumption
# component.gasConsumption
# component.heatGeneration
# component.capex
# component.CRF

import cvxpy as cp

class system:

    def __init__(self, name, components=None, timeIndex=None, powerLoad=None, heatLoad=None, powerPrice=None, gasPrice=None, powerMarginalEmissions=None, gasMarginalEmissions=None):
        self.name = name
        # list of components
        self.components = components
        # time series data
        self.timeIndex = timeIndex
        self.powerLoad = powerLoad
        self.heatLoad = heatLoad
        self.powerPrice = powerPrice
        self.gasPrice = gasPrice
        self.powerMarginalEmissions = powerMarginalEmissions
        self.gasMarginalEmissions = gasMarginalEmissions
        # system wide variables
        self.powerConsumption = None
        self.gasConsumption = None
        self.heatGeneration = None
        self.annnualizedCapex = None
        self.powerOpex = None
        self.gasOpex = None
        self.totalCost = None
        self.powerEmissions = None
        self.gasEmissions = None
        self.totalEmissions = None
        # optimization model
        self._variables = None
        self._constraints = None
        self._objective = None
        self._model = None
        

    def setTimeIndex(self, timeIndex):
        self.timeIndex = timeIndex

    def setPowerLoad(self, powerLoad):
        self.powerLoad = powerLoad
    
    def setHeatLoad(self, heatLoad):
        self.heatLoad = heatLoad
    
    def setPowerPrice(self, powerPrice):
        self.powerPrice = powerPrice
    
    def setGasPrice(self, gasPrice):
        self.gasPrice = gasPrice
    
    def setPowerEmissions(self, powerEmissions):
        self.powerEmissions = powerEmissions
    
    def setGasEmissions(self, gasEmissions):
        self.gasEmissions = gasEmissions
    
    def setTimeSeries(self, powerLoad, heatLoad, powerPrice, gasPrice, powerEmissions, gasEmissions):
        self.powerLoad = powerLoad
        self.heatLoad = heatLoad
        self.powerPrice = powerPrice
        self.gasPrice = gasPrice
        self.powerEmissions = powerEmissions
        self.gasEmissions = gasEmissions
    
    def addComponent(self, component):
        if self.components is None:
            self.components = []
        self.components.append(component)
    
    def getComponent(self, name):
        for component in self.components:
            if component.name == name:
                return component
        return None
    
    def _build_model(self, objective, emissionsCap=None, costCap=None):
        # initialize variables and constraints
        self._variables = []
        self._constraints = []
        self.powerConsumption = self.powerLoad
        self.gasConsumption = 0
        self.heatGeneration = 0
        self.annnualizedCapex = 0
        # add components variables and constraints
        # add components power and gas consumption and heat generation
        # add components capex
        for component in self.components:
            self._variables += component._variables
            self._constraints += component._constraints # Inner components constraints
            self.powerConsumption += component.powerConsumption
            self.gasConsumption += component.gasConsumption
            self.heatGeneration += component.heatGeneration
            self.annnualizedCapex += component.capex*component.CRF
        # add system wide constraints
        self._constraints += [self.powerConsumption >= 0]
        self._constraints += [self.gasConsumption  >= 0]
        self._constraints += [self.heatGeneration >= 0]
        self._constraints += [self.heatGeneration == self.heatLoad]
        # add system wide variables
        self.powerOpex = cp.pos(self.powerConsumption) @ self.powerPrice
        self.gasOpex = cp.pos(self.gasConsumption) @ self.gasPrice
        self.totalCost = self.powerOpex + self.gasOpex + self.annnualizedCapex
        self.powerEmissions = cp.pos(self.powerConsumption) @ self.powerMarginalEmissions
        self.gasEmissions = cp.pos(self.gasConsumption) @ self.gasMarginalEmissions
        self.totalEmissions = self.powerEmissions + self.gasEmissions
        # set objective
        if objective == 'cost':
            # minimize cost subject to total emissions cap
            self._objective = cp.Minimize(self.totalCost)
            if emissionsCap is not None:
                self._constraints += [self.totalEmissions <= emissionsCap]
        elif objective == 'emissions':
            # minimize total emissions subject to cost cap
            self._objective = cp.Minimize(self.totalEmissions)
            if costCap is not None:
                self._constraints += [self.totalCost <= costCap]
        # build model
        self._model = cp.Problem(self._objective, self._constraints)
    
    def solve(self, objective='cost', emissionsCap=None, costCap=None, solver=cp.CLARABEL, verbose=False):
        self._build_model(objective, emissionsCap, costCap)
        self._model.solve(solver=solver, verbose=verbose)
        return self._model.status
    



