
# from components : variables, inner constraints (like the state of charge and max delivered power, 
#but not the must meet load constraint, that's a system wide constraint) and gas and power input (power input can be negative) + heat output

# component.name
# component._variables
# component._constraints
# component.powerConsumption can be pos or neg
# component.gasConsumption 
# component.heatOutput can be pos or neg
# component.capex
# coponent.setOpex() à appeler depuis ici une fois le modèle résolu
# component.CRF

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
        self.LCOE = None
        self.LCOH = None
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
        self._constraints += [self.powerConsumption >= 0] # power load is met
        self._constraints += [self.gasConsumption  >= 0]
        self._constraints += [self.heatOutput >= 0]
        self._constraints += [self.heatOutput == self.heatLoad] # heat load is met
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
    
    def describe(self):
        raise NotImplementedError
    
    def plot(self):
        raise NotImplementedError
    
    def compare(self):
        raise NotImplementedError


class component:

    def __init__(self, name, parameters=None, variables=None, constraints=None, powerConsumption=None, gasConsumption=None, heatOutput=None, capex=None):
        self.name = name
        self._parameters = parameters
        self._variables = variables
        self._constraints = constraints
        self.powerConsumption = powerConsumption
        self.gasConsumption = gasConsumption
        self.heatOutput = heatOutput
        self.capex = capex
        self.CRF = None
        self.opex = None
    
    def describe(self):
        raise NotImplementedError
    
    
class NaturalGasFurnace(component):

    def __init__(self, n_timesteps=None, dt=None, capacityPrice=None, eff=None):
        '''
        Inputs:
            - n_timesteps: number of time steps
            - dt: interval between time steps in hours
            - capacityPrice: price of capacity in $/kW
            - eff: efficiency of the furnace in %
        '''

        name = 'NaturalGasFurnace'
        # Parameters
        parameters = {'capacityPrice': capacityPrice, 'eff': eff}
        # Variables
        gasInput = cp.Variable(n_timesteps, nonneg=True) # kWh
        capacity = cp.Variable(nonneg=True) # kW
        variables = [gasInput, capacity]
        # Derived quantities
        heatOutput = eff*gasInput # kWh
        # Constraints
        constraints = [heatOutput <= capacity*dt]
        # Consumption
        powerConsumption = 0
        gasConsumption = gasInput
        # Cost
        capex = capacity * capacityPrice # $

        super().__init__(name, parameters, variables, constraints, powerConsumption, gasConsumption, heatOutput, capex)

    # Check if gas consumption is in kWh or in m3.


class HeatPump(component):

    def __init__(self, n_timesteps=None, dt=None, COP=None, capacityPrice=None):
        '''
        Inputs:
            - n_timesteps: number of time steps
            - dt: interval between time steps in hours
            - COP: coefficient of performance of the heat pump
            - capacityPrice: price of capacity in $/kW
        '''

        name = 'HeatPump'
        # Parameters
        parameters = {'capacityPrice': capacityPrice, 'COP': COP}
        # Variables
        powerInput = cp.Variable(n_timesteps, nonneg=True) # kWh
        capacity = cp.Variable(nonneg=True) # kW
        variables = [powerInput, capacity]
        # Derived quantities
        heatOutput = COP * powerInput # kWh
        # Constraints
        constraints = [heatOutput <= capacity * dt]
        # Consumption
        powerConsumption = powerInput
        gasConsumption = 0
        # Cost
        capex = capacity * capacityPrice # $

        super().__init__(name, parameters, variables, constraints, powerConsumption, gasConsumption, heatOutput, capex)


class Battery(component):

    def __init__(self, n_timesteps=None, dt=None, socMin=None, socMax=None, socInitial=None, socFinal=None, 
                 maxDischargeRate=None, maxChargeRate=None, capacityPrice=None):
        '''
        Inputs:
            - n_timesteps: number of time steps
            - dt: interval between time steps in hours
            - socMin: minimum state of charge in % of battery capacity
            - socMax: maximum state of charge in % of battery capacity
            - socInitial: initial state of charge in % of battery capacity
            - socFinal: final state of charge in % of battery capacity
            - maxDischargeRate: maximum discharge rate in % of battery capacity
            - maxChargeRate: maximum charge rate in % of battery capacity
            - capacityPrice: price of capacity in $/kWh
        NB: State of charge soc is in kWh.
        '''

        name = 'Battery'
        # Parameters
        parameters = {'socMin': socMin, 'socMax': socMax, 'socInitial': socInitial, 'socFinal': socFinal, 
                      'maxDischargeRate': maxDischargeRate, 'maxChargeRate': maxChargeRate, 'capacityPrice': capacityPrice}
        # Variables
        powerInput = cp.Variable(n_timesteps) # kWh, positive when it charges, negative when it discharges
        soc = cp.Variable(n_timesteps, nonneg=True) # kWh
        energy_capacity = cp.Variable(nonneg=True) # kWh
        variables = [powerInput, soc, energy_capacity]
        # Constraints
        constraints = []
        constraints += [-powerInput <= maxDischargeRate * energy_capacity]
        constraints += [powerInput <= maxChargeRate * energy_capacity]
        constraints += [soc >= socMin * energy_capacity]
        constraints += [soc <= socMax * energy_capacity]
        constraints += [soc[0] == socInitial * energy_capacity]
        constraints += [soc[-1] == socFinal * energy_capacity]
        for t in range(n_timesteps-1):
            constraints += [soc[t+1] == soc[t] + powerInput[t]]
        # Consumption
        powerConsumption = powerInput # positive consumption (cost added) when it charges, negative (cost avoided) when it discharges
        gasConsumption = 0
        heatOutput = 0
        # Cost
        capex = energy_capacity * capacityPrice # $

        super().__init__(name, parameters, variables, constraints, powerConsumption, gasConsumption, heatOutput, capex)

        # TODO: check if need another variable for the powerAvailable??


class ThermalStorage(component):

    def __init__(self, n_timesteps=None, dt=None, socMin=None, socMax=None, socInitial=None, socFinal=None, 
                 maxDischargeRate=None, maxChargeRate=None, capacityPrice=None, lossRate=None):
        '''
        Inputs:
            - n_timesteps: number of time steps
            - dt: interval between time steps in hours
            - socMin: minimum state of charge in % of battery capacity
            - socMax: maximum state of charge in % of battery capacity
            - socInitial: initial state of charge in % of battery capacity
            - socFinal: final state of charge in % of battery capacity
            - maxDischargeRate: maximum discharge rate in % of battery capacity
            - maxChargeRate: maximum charge rate in % of battery capacity
            - capacityPrice: price of capacity in $/kWh
            - lossRate: rate of energy loss per hour in % of battery capacity
        NB: State of charge soc is in kWh.
        '''

        name = 'ThermalStorage'
        # Parameters
        parameters = {'socMin': socMin, 'socMax': socMax, 'socInitial': socInitial, 'socFinal': socFinal, 
                      'maxDischargeRate': maxDischargeRate, 'maxChargeRate': maxChargeRate, 'capacityPrice': capacityPrice, 'lossRate': lossRate}
        # Variables
        heatInput = cp.Variable(n_timesteps) # kWh, positive when it charges, negative when it discharges
        soc = cp.Variable(n_timesteps, nonneg=True) # kWh
        energy_capacity = cp.Variable(nonneg=True) # kWh
        variables = [heatInput, soc, energy_capacity]
        # Constraints
        constraints = []
        constraints += [-heatInput <= maxDischargeRate * energy_capacity]
        constraints += [heatInput <= maxChargeRate * energy_capacity]
        constraints += [soc >= socMin * energy_capacity]
        constraints += [soc <= socMax * energy_capacity]
        constraints += [soc[0] == socInitial * energy_capacity]
        constraints += [soc[-1] == socFinal * energy_capacity]
        for t in range(n_timesteps-1):
            constraints += [soc[t+1] == soc[t] + heatInput[t] - lossRate * dt * energy_capacity]
        # Consumption
        powerConsumption = 0
        gasConsumption = 0
        heatOutput = - heatInput # load added when it charges (heatInput positive), load avoided when it discharges (heatInput negative)
        # Cost
        capex = energy_capacity * capacityPrice # $

        super().__init__(name, parameters, variables, constraints, powerConsumption, gasConsumption, heatOutput, capex)

        # TODO: check if loss rate in % of capacity or current energy stored.


class PVsystem(component):

    def __init__(self, n_timesteps=None, dt=None, pvLoad=None, capacityPrice=None):
        '''
        Inputs:
            - n_timesteps: number of time steps
            - dt: interval between time steps in hours
            - pvLoad: time-indexed electricity available from the PV system, in % of the PV capacity
            - capacityPrice: price of capacity in $/kW
        '''

        name = 'PVsystem'
        # Parameters
        parameters = {'capacityPrice': capacityPrice, 'pvLoad': pvLoad}
        # Variables
        capacity = cp.Variable(nonneg=True) # kW
        variables = [capacity]
        # Derived quantities
        powerOutput = pvLoad * capacity * dt # kWh
        # Constraints
        constraints = []
        # Consumption
        powerConsumption = - powerOutput
        gasConsumption = 0
        heatOutput = 0
        # Cost
        capex = capacity * capacityPrice # $

        super().__init__(name, parameters, variables, constraints, powerConsumption, gasConsumption, heatOutput, capex)

    # Question: consider PV to be sold back to the grid?
