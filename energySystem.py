

# from components : variables, inner constraints (like the state of charge and max delivered power, 
#but not the must meet load constraint, that's a system wide constraint) and gas and power input (power input can be negative) + heat output

# component.name
# component._variables
# component._constraints
# component.powerConsumption can be pos or neg
# component.gasConsumption 
# component.heatOutput can be pos or neg
# component.capex
# component.setOpex() à appeler depuis ici une fois le modèle résolu
# component.CRF
# component.describe()

# TODO: check units

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
        self.heatOutput = None
        self.annnualizedCapex = None
        self.powerOpex = None
        self.gasOpex = None
        self.totalCost = None
        self.powerEmissions = None
        self.gasEmissions = None
        self.totalEmissions = None
        self.LCOE = None # TODO: how do we define LCOE / LCOH for a system?
        self.LCOH = None # TODO: how do we define LCOE / LCOH for a system?
        # optimization model
        self._variables = None
        self._constraints = None
        self._objective = None
        self._model = None
        self._status = None
        

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
        self.heatOutput = 0
        self.annnualizedCapex = 0
        # add components variables and constraints
        # add components power and gas consumption and heat generation
        # add components capex
        for component in self.components:
            self._variables += component._variables
            self._constraints += component._constraints # Inner components constraints
            self.powerConsumption += component.powerConsumption
            self.gasConsumption += component.gasConsumption
            self.heatOutput += component.heatOutput
            self.annnualizedCapex += component.capex*component.CRF
        # add system wide constraints
        self._constraints += [self.powerConsumption >= 0]
        self._constraints += [self.gasConsumption  >= 0]
        self._constraints += [self.heatOutput >= 0]
        self._constraints += [self.heatOutput == self.heatLoad]
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
    
    def _computeMetrics(self):
        raise NotImplementedError
    
    def solve(self, objective='cost', emissionsCap=None, costCap=None, solver=cp.CLARABEL, verbose=False):
        self._build_model(objective, emissionsCap, costCap)
        self._model.solve(solver=solver, verbose=verbose)
        self._status = self._model.status
        for component in self.components:
            component.setOpex()
        self._computeMetrics()
        return self._model.status
    
    def describe(self, detailed=False):
        print(f"System: {self.name}")
        print(f"{len(self.components)} components")
        for c in self.components:
            c.describe()
        print(f"Status: {self._status}")
        if self._status is "optimal":
            print(f"Total power consumption: {self.powerConsumption.value.sum()} kWh")
            print(f"Total gas consumption: {self.gasConsumption.value.sum()} kWh")
            print(f"Total cost: {self.totalCost.value} $")
            print(f"Total emissions: {self.totalEmissions.value} kgCO2")
            print(f"LCOE: {self.LCOE} $/kWh")
            print(f"LCOH: {self.LCOH} $/kWh")
        if detailed:
            print(f"Runs from {self.timeIndex[0]} to {self.timeIndex[-1]}")
            print(f"Annual power load: {self.powerLoad.sum()} kWh")
            print(f"Annual heat load: {self.heatLoad.sum()} kWh")
            print(f"Average power price: {self.powerPrice.sum()} $/kWh")
            print(f"Average gas price: {self.gasPrice.sum()} $/kWh")
            print(f"Average power emissions: {self.powerEmissions.sum()} kgCO2/kWh")
            print(f"Average gas emissions: {self.gasEmissions.sum()} kgCO2/kWh")
            if self._status is "optimal":
                print(f"Annualized capex: {self.annnualizedCapex.value} $")
                print(f"Power opex: {self.powerOpex.value} $")
                print(f"Gas opex: {self.gasOpex.value} $")
                print(f"Total power emissions: {self.powerEmissions.value.sum()} kgCO2")
                print(f"Total gas emissions: {self.gasEmissions.value.sum()} kgCO2")
    
    def plot(self):
        raise NotImplementedError
    
    def compare(self):
        raise NotImplementedError

    



