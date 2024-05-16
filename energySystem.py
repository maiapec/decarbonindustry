
# Component.describe()

### System
# TODO: find a better way to assign costs (CAPEX)
# TODO: heatmaps : adjust scale of cmap
# TODO: when adding components to a system, could we infer some parameters (like discount rate, ntimesteps, etc)
# TODO: be able to delete components from a system

### Component
# TODO: add efficiency to the battery and thermal storage
# TODO: make a dict of variables
# TODO: plots
# TODO: implement detailed or not in the describe method for components
# TODO: I changed the orders of the parameters in the __init__ of the components, check the string desciption (especially battery and thermal storage)
# TODO: add an option to set the parameters automatically from default values 


import cvxpy as cp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def getValue(variable):
    if isinstance(variable, cp.Variable):
        return variable.value
    elif isinstance(variable, cp.Expression):
        return variable.value
    else:
        return variable

class System:

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
        self.annualizedCapex = None
        self.powerOpex = None
        self.gasOpex = None
        self.totalCost = None
        self.powerEmissions = None
        self.gasEmissions = None
        self.totalEmissions = None
        self.LCOE = None
        self.LCOH = None
        self.CIE = None # Carbon Intensity of Electricity
        self.CIH = None # Carbon Intensity of Heat
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
        self.annualizedCapex = 0
        # add components variables and constraints
        # add components power and gas consumption and heat generation
        # add components capex
        for component in self.components:
            self._variables += component._variables
            self._constraints += component._constraints # Inner components constraints
            self.powerConsumption = self.powerConsumption + component.powerConsumption
            self.gasConsumption = self.gasConsumption + component.gasConsumption
            self.heatOutput = self.heatOutput + component.heatOutput
            self.annualizedCapex = self.annualizedCapex + component.capex*component.CRF
        # add system wide constraints
        self._constraints += [self.gasConsumption  >= 0]
        self._constraints += [self.heatOutput >= 0]
        self._constraints += [self.heatOutput == self.heatLoad] # heat load is met
        # add system wide variables
        self.powerOpex = cp.pos(self.powerConsumption) @ self.powerPrice
        self.gasOpex = cp.pos(self.gasConsumption) @ self.gasPrice
        self.totalCost = self.powerOpex + self.gasOpex + self.annualizedCapex
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
        # problem : how to assign power and gas consumption to heat and electricity consumption
        # especially if there is storage
        # one approach
        # 1. assign all gas cost to heat generation
        # 2. compute alpha = annual power load (as power) / annual power consumption
        #    assign alpha * power cost to power load and (1-alpha) * power cost to heat generation
        # 3. LCOH = (gas cost + (1-alpha) * power cost) / annual heat load
        # 4. LCOE = alpha * power cost / annual power load
        # What about CAPEX ? could assign using the same method
        # Not very satisfying, heat production device should only be assigned to heat
        # But for now will do
        # Same approach for emissions
        if self._status != "optimal":
            print("Model not solved")
        else:
            pwrCons = getValue(self.powerConsumption)
            alpha = self.powerLoad.sum() / pwrCons.sum()
            self.LCOH = (self.gasOpex.value + (1-alpha) * self.powerOpex.value + (1-alpha)*self.annualizedCapex) / self.heatLoad.sum()
            self.LCOE = (alpha * self.powerOpex.value + alpha*self.annualizedCapex) / self.powerLoad.sum()
            self.CIH = (self.gasEmissions.value + (1-alpha) * self.powerEmissions.value) / self.heatLoad.sum()
            self.CIE = alpha * self.powerEmissions.value / self.powerLoad.sum()
    
    def solve(self, objective='cost', emissionsCap=None, costCap=None, solver=cp.CLARABEL, verbose=False):
        self._build_model(objective, emissionsCap, costCap)
        self._model.solve(solver=solver, verbose=verbose)
        self._status = self._model._status
        for component in self.components:
            component.setOpex(self.powerPrice, self.gasPrice)
        self._computeMetrics()
        return self._model._status
    
    def describe(self, detailed=False):
        print(f"System: {self.name}")
        print(f"{len(self.components)} component(s)")
        for c in self.components:
            c.describe()
        print(f"Status: {self._status}")
        print("")
        if self._status == "optimal":
            pwrCons = getValue(self.powerConsumption)
            gasCons = getValue(self.gasConsumption)
            print(f"Annual power consumption: {np.round(pwrCons.sum()/1000)} MWh")
            print(f"Annual gas consumption: {np.round(gasCons.sum()/1000)} kWh")
            print(f"Annual cost: {np.round(self.totalCost.value/1e6, 3)} M$")
            print(f"Annual emissions: {np.round(self.totalEmissions.value/1e6, 2)} MtonCO2")
            print(f"LCOE (Electricity): {np.round(self.LCOE.value, 3)} $/kWh")
            print(f"LCOH (Heat): {np.round(self.LCOH.value, 3)} $/kWh")
            print(f"Carbon Intensity of Electricity: {np.round(self.CIE, 3)} kgCO2/kWhe")
            print(f"Carbon Intensity of Heat: {np.round(self.CIH, 3)} kgCO2/kWhth")
        print("")
        if detailed:
            print(f"Runs from {self.timeIndex[0]} to {self.timeIndex[-1]}")
            print(f"Annual power load: {np.round(self.powerLoad.sum()/1000)} MWh")
            print(f"Annual heat load: {np.round(self.heatLoad.sum()/1000)} MWh")
            print(f"Average supply power price: {np.round(self.powerPrice.mean(), 3)} $/kWh")
            print(f"Average supply gas price: {np.round(self.gasPrice.mean(), 3)} $/kWh")
            print(f"Average supply power emissions: {np.round(self.powerMarginalEmissions.mean(), 3)} kgCO2/kWhe")
            print(f"Average supply gas emissions: {np.round(self.gasMarginalEmissions.mean(), 3)} kgCO2/kWhth")
            if self._status == "optimal":
                print(f"Annualized capex: {np.round(self.annualizedCapex.value/1e6, 3)} M$")
                print(f"Power opex: {np.round(self.powerOpex.value/1e6, 3)} M$")
                print(f"Gas opex: {np.round(self.gasOpex.value/1e6, 3)} M$")
                print(f"Total power emissions: {np.round(self.powerEmissions.value.sum()/1e6, 2)} MtonCO2")
                print(f"Total gas emissions: {np.round(self.gasEmissions.value.sum()/1e6, 2)} MtonCO2")
    
    def plot(self):
        raise NotImplementedError

    def _pivot(self, timeSeries):
        df = pd.DataFrame(timeSeries, index=self.timeIndex)
        pivot_df = df.pivot_table(index=df.index.time, columns=df.index.date)
        pivot_df.columns = pivot_df.columns.droplevel()
        return pivot_df

    
    def plotHeatmaps(self):
        nc = len(self.components)
        fig, axs = plt.subplots(nc+2, 2, figsize=(15, 4*nc), dpi=300, sharex='col', sharey='row')
        cmap = 'coolwarm'
        # Power Load
        sns.heatmap(self._pivot(self.powerLoad), ax=axs[0, 0], cmap=cmap, cbar_kws={'label': 'kWhe'})
        axs[0, 0].set_title('Power Load')
        axs[0, 0].set_xlabel('')
        axs[0, 0].set_ylabel('Time')
        # Heat Load
        sns.heatmap(self._pivot(self.heatLoad), ax=axs[0, 1], cmap=cmap, cbar_kws={'label': 'kWhth'})
        axs[0, 1].set_title('Heat Load')
        axs[0, 1].set_xlabel('')
        axs[0, 1].set_ylabel('')
        # Components
        for c in self.components:
            i = self.components.index(c) + 1
            # Power Consumption
            sns.heatmap(self._pivot(getValue(c.powerConsumption)), ax=axs[i, 0], cmap=cmap, cbar_kws={'label': 'kWhe'})
            axs[i, 0].set_title(f'{c.name} Power Consumption')
            axs[i, 0].set_xlabel('')
            axs[i, 0].set_ylabel('Time')
            # Heat Output
            sns.heatmap(self._pivot(getValue(c.heatOutput)), ax=axs[i, 1], cmap=cmap, cbar_kws={'label': 'kWhth'})
            axs[i, 1].set_title(f'{c.name} Heat Output')
            axs[i, 1].set_xlabel('')
            axs[i, 1].set_ylabel('')
        # Total Consumptions
        sns.heatmap(self._pivot(getValue(self.powerConsumption)), ax=axs[-1, 0], cmap=cmap, cbar_kws={'label': 'kWhe'})
        axs[-1, 0].set_title('Total Power Consumption')
        axs[-1, 0].set_xlabel('Date')
        axs[-1, 0].set_ylabel('Time')
        axs[-1, 0].set_xticklabels(axs[-1, 0].get_xticklabels(), rotation=60)
        sns.heatmap(self._pivot(getValue(self.gasConsumption)), ax=axs[-1, 1], cmap=cmap, cbar_kws={'label': 'kWhgas'})
        axs[-1, 1].set_title('Total Gas Consumption')
        axs[-1, 1].set_xlabel('Date')
        axs[-1, 1].set_ylabel('')
        axs[-1, 1].set_xticklabels(axs[-1, 1].get_xticklabels(), rotation=60)
        plt.tight_layout()
        return plt.gca()
    
    def compare(self):
        raise NotImplementedError

class Component:

    def __init__(self, name, parameters=None, variables=None, constraints=None, powerConsumption=None, gasConsumption=None, heatOutput=None, capex=None, CRF=None):
        self.name = name
        self._parameters = parameters
        self._variables = variables
        self._constraints = constraints
        self.powerConsumption = powerConsumption
        self.gasConsumption = gasConsumption
        self.heatOutput = heatOutput
        self.capex = capex
        self.CRF = CRF
        self.opex = None
    
    def describe(self):
        print(f"Component: {self.name}")
        if self._parameters is not None:
            for k, v in self._parameters.items():
                print(f"    {k}: {v}")
    
    def setOpex(self, powerPrice, gasPrice):
        if self.opex is None:
            pass
        pwrCons = getValue(self.powerConsumption)
        gasCons = getValue(self.gasConsumption)
        self.opex = pwrCons @ powerPrice + gasCons @ gasPrice
    
    
class NaturalGasFurnace(Component):

    def __init__(self, n_timesteps=None, dt=None, eff=None, capacityPrice=None, discRate=None, n_years=None):
        '''
        Inputs:
            - n_timesteps: number of time steps
            - dt: interval between time steps in hours
            - capacityPrice: price of capacity in $/kW
            - eff: efficiency of the furnace in %
            - discRate: discount rate in %
            - n_years: lifetime of the Component in years
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
        powerConsumption = np.zeros(n_timesteps)
        gasConsumption = gasInput
        # Cost
        capex = capacity * capacityPrice # $
        CRF = discRate * (1 + discRate)**n_years / ((1 + discRate)**n_years - 1)

        super().__init__(name, parameters, variables, constraints, powerConsumption, gasConsumption, heatOutput, capex, CRF)

        # Store specific attributes
        self.gasInput = gasInput
        self.capacity = capacity

    def describe(self):
        print(f"Component: {self.name}")
        if self._parameters is not None:
            for k, v in self._parameters.items():
                print(f"    {k}: {v}")
        print(f"    Optimal capacity: {np.round(self.capacity.value)} kW")


class HeatPump(Component):

    def __init__(self, n_timesteps=None, dt=None, COP=None, capacityPrice=None, discRate=None, n_years=None):
        '''
        Inputs:
            - n_timesteps: number of time steps
            - dt: interval between time steps in hours
            - COP: coefficient of performance of the heat pump
            - capacityPrice: price of capacity in $/kW
            - discRate: discount rate in %
            - n_years: lifetime of the component in years
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
        gasConsumption = np.zeros(n_timesteps)
        # Cost
        capex = capacity * capacityPrice # $
        CRF = discRate * (1 + discRate)**n_years / ((1 + discRate)**n_years - 1)

        super().__init__(name, parameters, variables, constraints, powerConsumption, gasConsumption, heatOutput, capex, CRF)

        # Store specific attributes
        self.powerInput = powerInput
        self.capacity = capacity

    def describe(self):
        print(f"Component: {self.name}")
        if self._parameters is not None:
            for k, v in self._parameters.items():
                print(f"    {k}: {v}")
        print(f"    Optimal capacity: {np.round(self.capacity.value)} kW")


class Battery(Component):

    def __init__(self, n_timesteps=None, dt=None, maxChargeRate=None, capacityPrice=None,
                 maxDischargeRate=None, socMin=None, socMax=None, socInitial=None, socFinal=None, 
                 discRate=None, n_years=None, name="Battery"):
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
            - discRate: discount rate in %
            - n_years: lifetime of the component in years
        NB: State of charge soc is in kWh.
        '''

        if maxChargeRate is not None:
            if maxDischargeRate is None:
                maxDischargeRate = maxChargeRate
            if socMin is None:
                socMin = 0
            if socMax is None:
                socMax = 1
            if socInitial is None:
                socInitial = 0.5
            if socFinal is None:
                socFinal = 0.5

        # Parameters
        parameters = {'socMin': socMin, 'socMax': socMax, 'socInitial': socInitial, 'socFinal': socFinal, 
                      'maxDischargeRate': maxDischargeRate, 'maxChargeRate': maxChargeRate, 'capacityPrice': capacityPrice}
        # Variables
        powerInput = cp.Variable(n_timesteps) # kWh, positive when it charges, negative when it discharges
        soc = cp.Variable(n_timesteps+1, nonneg=True) # kWh
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
        constraints += [soc[1:] == soc[:-1] + powerInput] # roughly 15 times quicker to solve when vectorized
        # Consumption
        powerConsumption = powerInput # positive consumption (cost added) when it charges, negative (cost avoided) when it discharges
        gasConsumption = np.zeros(n_timesteps)
        heatOutput = np.zeros(n_timesteps)
        # Cost
        capex = energy_capacity * capacityPrice # $
        CRF = discRate * (1 + discRate)**n_years / ((1 + discRate)**n_years - 1)

        super().__init__(name, parameters, variables, constraints, powerConsumption, gasConsumption, heatOutput, capex, CRF)

        # Store specific attributes
        self.powerInput = powerInput
        self.soc = soc
        self.energy_capacity = energy_capacity

    def describe(self):
        print(f"Component: {self.name}")
        if self._parameters is not None:
            for k, v in self._parameters.items():
                print(f"    {k}: {v}")
        print(f"    Optimal energy capacity: {np.round(self.energy_capacity.value)} kWh")
        print(f"    Optimal power capacity: {np.round(self._parameters['maxChargeRate'] * self.energy_capacity.value)} kW")


# Should we optimize for C rate
class ThermalStorage(Component):

    def __init__(self, n_timesteps=None, dt=None, maxChargeRate=None, lossRate=None, capacityPrice=None,
                 maxDischargeRate=None, socMin=None, socMax=None, socInitial=None, socFinal=None, 
                 discRate=None, n_years=None):
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
            - discRate: discount rate in %
            - n_years: lifetime of the component in years
        NB: State of charge soc is in kWh.
        '''

        if maxChargeRate is not None:
            if maxDischargeRate is None:
                maxDischargeRate = maxChargeRate
            if socMin is None:
                socMin = 0
            if socMax is None:
                socMax = 1
            if socInitial is None:
                socInitial = 0.5
            if socFinal is None:
                socFinal = 0.5

        name = 'ThermalStorage'
        # Parameters
        parameters = {'socMin': socMin, 'socMax': socMax, 'socInitial': socInitial, 'socFinal': socFinal, 
                      'maxDischargeRate': maxDischargeRate, 'maxChargeRate': maxChargeRate, 'capacityPrice': capacityPrice, 'lossRate': lossRate}
                      
        # Variables
        heatInput = cp.Variable(n_timesteps) # kWh, positive when it charges, negative when it discharges
        soc = cp.Variable(n_timesteps + 1, nonneg=True) # kWh
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
        constraints += [soc[1:] == soc[:-1]*(1 - dt*lossRate) + heatInput]
        # Consumption
        powerConsumption = np.zeros(n_timesteps)
        gasConsumption = np.zeros(n_timesteps)
        heatOutput = - heatInput # load added when it charges (heatInput positive), load avoided when it discharges (heatInput negative)
        # Cost
        capex = energy_capacity * capacityPrice # $
        CRF = discRate * (1 + discRate)**n_years / ((1 + discRate)**n_years - 1)

        super().__init__(name, parameters, variables, constraints, powerConsumption, gasConsumption, heatOutput, capex, CRF)

        # TODO: Maya : check if loss rate in % of capacity or current energy stored.
        # Aramis : I think it is in % of current energy stored, (because it depends on the temperature of the storage, which is related to the energy stored)

        # Store specific attributes
        self.heatInput = heatInput
        self.soc = soc
        self.energy_capacity = energy_capacity
    
    def describe(self):
        print(f"Component: {self.name}")
        if self._parameters is not None:
            for k, v in self._parameters.items():
                print(f"    {k}: {v}")
        print(f"    Optimal energy capacity: {np.round(self.energy_capacity.value)} kWh")
        print(f"    Optimal power capacity: {np.round(self._parameters['maxChargeRate'] * self.energy_capacity.value)} kW")


class PVsystem(Component):

    def __init__(self, n_timesteps=None, dt=None, pvLoad=None, capacityPrice=None,  discRate=None, n_years=None):
        '''
        Inputs:
            - n_timesteps: number of time steps
            - dt: interval between time steps in hours
            - pvLoad: time-indexed electricity available from the PV system, in % of the PV capacity
            - capacityPrice: price of capacity in $/kW
            - discRate: discount rate in %
            - n_years: lifetime of the component in years
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
        CRF = discRate * (1 + discRate)**n_years / ((1 + discRate)**n_years - 1)

        super().__init__(name, parameters, variables, constraints, powerConsumption, gasConsumption, heatOutput, capex, CRF)

        # Store specific attributes
        self.capacity = capacity

    # Question: consider PV to be sold back to the grid?
