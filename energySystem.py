# Component.describe()

### System
# TODO: solar profile is for 2022, change marginal emissions ?
# TODO: find a better way to assign costs (CAPEX)
# TODO: heatmaps : adjust scale of cmap
# TODO: be able to delete components from a system
# TODO: add demand charge
# TODO: recompute LCOE, LCOH, emissions per kwh when there is PV 
# TODO: solve with carbon price

### Component
# TODO: plots
# TODO: implement detailed or not in the describe method for components

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

    def __init__(self, name, components=None, timeIndex=None, powerLoad=None, heatLoad=None, powerPrice=None, sellBackPrice=None, powerDemandFee=None, gasPrice=None, gridMarginalEmissions=None, gasMarginalEmissions=None):
        self.name = name
        # list of components
        self.components = components
        # time series data
        self.timeIndex = timeIndex
        self.dt = (timeIndex[1] - timeIndex[0]).seconds / 3600
        self.powerLoad = powerLoad
        self.heatLoad = heatLoad
        self.powerPrice = powerPrice
        self.sellBackPrice = sellBackPrice
        self.powerDemandPrice = powerDemandFee
        self.gasPrice = gasPrice
        self.gridMarginalEmissions = gridMarginalEmissions
        self.gasMarginalEmissions = gasMarginalEmissions
        # system wide variables
        self.netPowerConsumption = None
        self.powerConsumption = None
        self.powerGeneration = None
        self.gasConsumption = None
        self.heatOutput = None
        self.annualizedCapex = None
        self.powerOpex = None
        self.gasOpex = None
        self.totalCost = None
        self.powerOperationalEmissions = None
        self.gasOperationalEmissions = None
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
    
    def setPowerOperationalEmissions(self, powerOperationalEmissions):
        self.powerOperationalEmissions = powerOperationalEmissions
    
    def setGasOperationalEmissions(self, gasOperationalEmissions):
        self.gasOperationalEmissions = gasOperationalEmissions
    
    def setTimeSeries(self, powerLoad, heatLoad, powerPrice, gasPrice, powerOperationalEmissions, gasOperationalEmissions):
        self.powerLoad = powerLoad
        self.heatLoad = heatLoad
        self.powerPrice = powerPrice
        self.gasPrice = gasPrice
        self.powerOperationalEmissions = powerOperationalEmissions
        self.gasOperationalEmissions = gasOperationalEmissions
    
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
        self.netPowerConsumption = self.powerLoad
        self.powerConsumption = self.powerLoad
        self.powerGeneration = 0
        self.gasConsumption = 0
        self.heatOutput = 0
        self.annualizedCapex = 0
        self.powerOperationalEmissions = 0
        # add components variables and constraints
        # add components power and gas consumption and heat generation
        # add components capex
        for component in self.components:
            self._variables += component._variables
            self._constraints += component._constraints # Inner components constraints
            self.netPowerConsumption = self.netPowerConsumption + component.powerConsumption
            if component.typeTransfer == 'ElectricityGeneration':
                self.powerGeneration = self.powerGeneration - component.powerConsumption
            elif component.typeTransfer == 'Battery':
                self.powerGeneration = self.powerGeneration + component._variablesDict['powerInputDischarge']
                self.powerConsumption = self.powerConsumption + component._variablesDict['powerInputCharge']
            else:
                self.powerConsumption = self.powerConsumption + component.powerConsumption
            self.gasConsumption = self.gasConsumption + component.gasConsumption
            self.heatOutput = self.heatOutput + component.heatOutput
            self.annualizedCapex = self.annualizedCapex + component.capex*component.CRF 
        # add system wide constraints
        self._constraints += [self.gasConsumption  >= 0]
        self._constraints += [self.heatOutput >= 0]
        self._constraints += [self.heatOutput == self.heatLoad] # heat load is met
        # add system wide variables
        self.powerOpex = cp.pos(self.netPowerConsumption) @ (self.powerPrice - self.sellBackPrice) + self.netPowerConsumption @ self.sellBackPrice
        # gridPower is the power seen by the meter, on which the demand charge is computed
        self.gridPower = self.netPowerConsumption
        for component in self.components:
            if component.typeTransfer == 'ElectricityGeneration' and not component._parameters['onsite']:
                self.gridPower = self.gridPower - component.powerConsumption
        self.powerOpex = self.powerOpex + np.sum([cp.max(self.gridPower[d[0]])*d[1] for d in self.powerDemandPrice])
        # d[0] is a mask for the demand periods (such as peak / offpeak), d[1] is the price
        self.gasOpex = cp.pos(self.gasConsumption) @ self.gasPrice
        self.totalCost = self.powerOpex + self.gasOpex + self.annualizedCapex
        self.powerOperationalEmissions = cp.pos(self.netPowerConsumption) @ self.gridMarginalEmissions
        self.gasOperationalEmissions = cp.pos(self.gasConsumption) @ self.gasMarginalEmissions
        self.totalEmissions = self.powerOperationalEmissions + self.gasOperationalEmissions
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
            pwrCons = getValue(self.netPowerConsumption)
            alpha = self.powerLoad.sum() / pwrCons.sum()
            self.LCOH = (self.gasOpex.value + (1-alpha) * self.powerOpex.value + (1-alpha)*self.annualizedCapex) / self.heatLoad.sum()
            self.LCOE = (alpha * self.powerOpex.value + alpha*self.annualizedCapex) / self.powerLoad.sum()
            self.CIH = (self.gasOperationalEmissions.value + (1-alpha) * self.powerOperationalEmissions.value) / self.heatLoad.sum()
            self.CIE = alpha * self.powerOperationalEmissions.value / self.powerLoad.sum()
    
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
            pwrCons = getValue(self.netPowerConsumption)
            gasCons = getValue(self.gasConsumption)
            print(f"Annual power consumption: {np.round(pwrCons.sum()/1000)} MWh")
            print(f"Annual gas consumption: {np.round(gasCons.sum()/1000)} MWh")
            print(f"Annual cost: {np.round(self.totalCost.value/1e6, 3)} M$")
            print(f"Annual operational emissions: {np.round(self.totalEmissions.value/1e6, 2)} MtonCO2")
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
            print(f"Average supply power emissions: {np.round(self.gridMarginalEmissions.mean(), 3)} kgCO2/kWhe")
            print(f"Average supply gas emissions: {np.round(self.gasMarginalEmissions.mean(), 3)} kgCO2/kWhth")
            if self._status == "optimal":
                print(f"Annualized capex: {np.round(self.annualizedCapex.value/1e6, 3)} M$")
                print(f"Power opex: {np.round(self.powerOpex.value/1e6, 3)} M$")
                print(f"Gas opex: {np.round(self.gasOpex.value/1e6, 3)} M$")
                print(f"Total power emissions: {np.round(self.powerOperationalEmissions.value.sum()/1e6, 2)} MtonCO2")
                print(f"Total gas emissions: {np.round(self.gasOperationalEmissions.value.sum()/1e6, 2)} MtonCO2")

    def _pivot(self, timeSeries):
        df = pd.DataFrame(timeSeries, index=self.timeIndex)
        pivot_df = df.pivot_table(index=df.index.time, columns=df.index.date)
        pivot_df.columns = pivot_df.columns.droplevel()
        return pivot_df
    
    def getPowerDataFrame(self):
        pwr = pd.DataFrame(self.powerLoad, index=self.timeIndex, columns=['Power Load'])
        for c in self.components:
            pwr[c.name] = getValue(c.powerConsumption)
        pwr['Total Power Consumption'] = getValue(self.netPowerConsumption)
        pwr = pwr / self.dt # kWh to kW
        pwr['Marginal Power Price'] = self.powerPrice
        pwr['Marginal Power Emissions'] = self.gridMarginalEmissions
        return pwr
    
    def getHeatDataFrame(self):
        heat = pd.DataFrame(self.heatLoad, index=self.timeIndex, columns=['Heat Load'])
        for c in self.components:
            heat[c.name] = getValue(c.heatOutput)
        heat['Total Heat Output'] = getValue(self.heatOutput)
        heat = heat / self.dt # kWh to kW
        return heat

    def plot_power(self, colors, powerConsumers, powerStorage, powerGenerators, period=None, start=None, end=None):
        pwr = self.getPowerDataFrame()
        # Group
        if period is not None and start is None and end is None:
            pwr = pwr.loc[period]
        elif period is None and start is not None and end is not None:
            pwr = pwr.loc[start:end]
        else:
            raise ValueError("Please provide either a period or a start and end date") 
        # Preprocess
        pwr_pos = pwr[powerConsumers].copy() 
        pwr_neg = pwr[powerGenerators].copy()
        pwr_batt_pos, pwr_battery_neg = pwr[powerStorage].clip(lower=0), pwr[powerStorage].clip(upper=0)
        pwr_pos[powerStorage] = pwr_batt_pos
        pwr_neg[''] = pwr_battery_neg
        # Plot
        fig, axs = plt.subplots(2, figsize=(15, 10), dpi=300, sharex=True)
        pwr_pos.plot.area(color=colors, ax=axs[0])
        pwr_neg.plot.area(color=colors, ax=axs[0])
        pwr['Total Power Consumption'].plot(color=colors, ax=axs[0])
        pwr['Marginal Power Price'].plot(color=colors['Marginal Power Price'], linestyle='--', ax=axs[1])
        pwr['Marginal Power Emissions'].plot(color=colors['Marginal Power Emissions'], linestyle='--', ax=axs[1], secondary_y=True)
        axs[0].set_ylabel('Power Consumption(kW)')
        axs[1].set_ylabel('Price ($/kWh)', color=colors['Marginal Power Price'])
        axs[1].right_ax.set_ylabel('Emissions (kgCO2/kWh)', color=colors['Marginal Power Emissions'])
        axs[0].legend()
        plt.tight_layout
        return plt.gca()
    
    def plot_heat(self, colors, heatGenerators, heatStorage, period=None, start=None, end=None):
        heat = self.getHeatDataFrame()
        power = self.getPowerDataFrame()
        # Group
        if period is not None and start is None and end is None:
            heat = heat.loc[period]
            power = power.loc[period]
        elif period is None and start is not None and end is not None:
            heat = heat.loc[start:end]
            power = power.loc[start:end]
        else:
            raise ValueError("Please provide either a period or a start and end date")
        # Preprocess
        heat_pos = heat[heatGenerators].copy()
        heat_sto_pos, heat_sto_neg = heat[heatStorage].clip(lower=0), heat[heatStorage].clip(upper=0)
        heat_pos[heatStorage] = heat_sto_pos
        # Plot
        fig, axs = plt.subplots(2, figsize=(15, 10), dpi=300, sharex=True)
        heat_pos.plot.area(color=colors, ax=axs[0])
        for heatSto in heatStorage:
            heat_sto_neg[heatSto].plot.area(color=colors[heatSto], label='', ax=axs[0])
        heat['Heat Load'].plot(color=colors['Heat Load'], ax=axs[0])
        power['Marginal Power Price'].plot(color=colors, linestyle='--', ax=axs[1])
        power['Marginal Power Emissions'].plot(color=colors, linestyle='--', ax=axs[1], secondary_y=True)
        axs[0].set_ylabel('Heat Generation(kW)')
        axs[1].set_ylabel('Price ($/kWh)', color=colors['Marginal Power Price'])
        axs[1].right_ax.set_ylabel('Emissions (kgCO2/kWh)', color=colors['Marginal Power Emissions'])
        axs[0].legend()
        plt.tight_layout
        return plt.gca()
    
    def plotHeatmaps(self):
        nc = len(self.components)
        fig, axs = plt.subplots(nc+2, 2, figsize=(15, 3*nc+2), dpi=300, sharex='col', sharey='row')
        cmap = 'coolwarm'
        # Power Load
        sns.heatmap(self._pivot(self.powerLoad/self.dt), ax=axs[0, 0], cmap=cmap, cbar_kws={'label': 'kWe'}) # kWh to kW
        axs[0, 0].set_title('Power Load')
        axs[0, 0].set_xlabel('')
        axs[0, 0].set_ylabel('Time')
        # Heat Load
        sns.heatmap(self._pivot(self.heatLoad/self.dt), ax=axs[0, 1], cmap=cmap, cbar_kws={'label': 'kWth'}) # kWh to kW
        axs[0, 1].set_title('Heat Load')
        axs[0, 1].set_xlabel('')
        axs[0, 1].set_ylabel('')
        # Components
        for c in self.components:
            i = self.components.index(c) + 1
            # Power Consumption
            if (c.name == 'PVsystem') or (c.name == 'Windsystem'):
                pwrCons = - self._pivot(getValue(c.powerConsumption)/self.dt) # kWh to kW
                ttl = c.name + ' Power Generation'
            else:
                pwrCons = self._pivot(getValue(c.powerConsumption)/self.dt) # kWh to kW
                ttl = f'{c.name} Power Consumption'
            sns.heatmap(pwrCons, ax=axs[i, 0], cmap=cmap, cbar_kws={'label': 'kWe'})
            axs[i, 0].set_title(ttl)
            axs[i, 0].set_xlabel('')
            axs[i, 0].set_ylabel('Time')
            # Heat Output
            sns.heatmap(self._pivot(getValue(c.heatOutput)/self.dt), ax=axs[i, 1], cmap=cmap, cbar_kws={'label': 'kWth'}) # kWh to kW
            axs[i, 1].set_title(f'{c.name} Heat Output')
            axs[i, 1].set_xlabel('')
            axs[i, 1].set_ylabel('')
        # Total Consumptions
        sns.heatmap(self._pivot(getValue(self.netPowerConsumption)/self.dt), ax=axs[-1, 0], cmap=cmap, cbar_kws={'label': 'kWe'}) # kWh to kW
        axs[-1, 0].set_title('Power bought from Grid')
        axs[-1, 0].set_xlabel('Date')
        axs[-1, 0].set_ylabel('Time')
        axs[-1, 0].set_xticklabels(axs[-1, 0].get_xticklabels(), rotation=60)
        sns.heatmap(self._pivot(getValue(self.gridPower)/self.dt), ax=axs[-1, 1], cmap=cmap, cbar_kws={'label': 'kWe'}) # kWh to kW
        axs[-1, 1].set_title('Power consumed from Grid')
        axs[-1, 1].set_xlabel('Date')
        axs[-1, 1].set_ylabel('')
        axs[-1, 1].set_xticklabels(axs[-1, 1].get_xticklabels(), rotation=60)
        plt.tight_layout()
        return plt.gca()
    
    def compare(self):
        raise NotImplementedError

class Component:

    def __init__(self, name, typeTransfer=None, parameters=None, variables=None, variablesDict=None, constraints=None, powerConsumption=None, gasConsumption=None, heatOutput=None, capex=None, CRF=None):
        self.name = name
        self.typeTransfer = typeTransfer
        self._parameters = parameters
        self._variables = variables
        self._variablesDict = variablesDict
        self._constraints = constraints
        self.powerConsumption = powerConsumption # kWh
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
            - discRate: discount rate
            - n_years: lifetime of the Component in years
        '''

        name = 'NaturalGasFurnace'
        typeTransfer = 'GasToHeat'
        # Parameters
        parameters = {'capacityPrice': capacityPrice, 'eff': eff}
        # Variables
        gasInput = cp.Variable(n_timesteps, nonneg=True) # kWh
        capacity = cp.Variable(nonneg=True) # kW
        variables = [gasInput, capacity]
        # Save variables in a dictionary
        variablesDict = {'gasInput': gasInput, 'capacity': capacity}
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

        super().__init__(name, typeTransfer, parameters, variables, variablesDict, constraints, powerConsumption, gasConsumption, heatOutput, capex, CRF)

    def describe(self):
        print(f"Component: {self.name}")
        if self._parameters is not None:
            for k, v in self._parameters.items():
                print(f"    {k}: {v}")
        print(f"    Optimal capacity: {np.round(self._variablesDict['capacity'].value)} kW")


class HeatPump(Component):

    def __init__(self, n_timesteps=None, dt=None, COP=None, capacityPrice=None, discRate=None, n_years=None):
        '''
        Inputs:
            - n_timesteps: number of time steps
            - dt: interval between time steps in hours
            - COP: coefficient of performance of the heat pump
            - capacityPrice: price of capacity in $/kW
            - discRate: discount rate
            - n_years: lifetime of the component in years
        '''

        name = 'HeatPump'
        typeTransfer = 'ElectricityToHeat'
        # Parameters
        parameters = {'capacityPrice': capacityPrice, 'COP': COP}
        # Variables
        powerInput = cp.Variable(n_timesteps, nonneg=True) # kWh
        capacity = cp.Variable(nonneg=True) # kW
        variables = [powerInput, capacity]
        # Save variables in a dictionary
        variablesDict = {'powerInput': powerInput, 'capacity': capacity}
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

        super().__init__(name, typeTransfer, parameters, variables, variablesDict, constraints, powerConsumption, gasConsumption, heatOutput, capex, CRF)

    def describe(self):
        print(f"Component: {self.name}")
        if self._parameters is not None:
            for k, v in self._parameters.items():
                print(f"    {k}: {v}")
        print(f"    Optimal capacity: {np.round(self._variablesDict['capacity'].value)} kW")


class Battery(Component):

    def __init__(self, n_timesteps=None, dt=None, capacityPrice=None, 
                 maxChargeRate=None, effCharge = None, effDischarge = None,
                 maxDischargeRate=None, socMin=None, socMax=None, socInitial=None, socFinal=None, 
                 discRate=None, n_years=None, name="Battery"):
        '''
        Inputs:
            - n_timesteps: number of time steps
            - dt: interval between time steps in hours
            - capacityPrice: price of capacity in $/kWh

            - maxChargeRate: maximum charge rate in % of battery capacity
            - effCharge: efficiency of the battery when charging
            - effDischarge: efficiency of the battery when discharging
            - maxDischargeRate: maximum discharge rate in % of battery capacity

            - socMin: minimum state of charge in % of battery capacity
            - socMax: maximum state of charge in % of battery capacity
            - socInitial: initial state of charge in % of battery capacity
            - socFinal: final state of charge in % of battery capacity
            
            - discRate: discount rate
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

        typeTransfer = 'Battery'
        # Parameters
        parameters = {'socMin': socMin, 'socMax': socMax, 'socInitial': socInitial, 'socFinal': socFinal,
                      'maxDischargeRate': maxDischargeRate, 'maxChargeRate': maxChargeRate, 'capacityPrice': capacityPrice,
                      'effCharge': effCharge, 'effDischarge': effDischarge}
        # Variables
        powerInputCharge = cp.Variable(n_timesteps) # kWh
        powerInputDischarge = cp.Variable(n_timesteps)
        powerInput = powerInputCharge - powerInputDischarge # kWh, positive when it charges, negative when it discharges
        soc = cp.Variable(n_timesteps+1, nonneg=True) # kWh
        energyCapacity = cp.Variable(nonneg=True) # kWh
        variables = [powerInputCharge, powerInputDischarge, soc, energyCapacity]
        # Save variables in a dictionary
        variablesDict = {'powerInput': powerInput, 'powerInputCharge': powerInputCharge, 'powerInputDischarge': powerInputDischarge,
                         'soc': soc, 'energyCapacity': energyCapacity}
        # Constraints
        constraints = []
        constraints += [-powerInput <= maxDischargeRate * dt * energyCapacity] # maxDischargeRate is defined for an hour
        constraints += [powerInput <= maxChargeRate * dt * energyCapacity] # maxChargeRate is defined for an hour
        constraints += [soc >= socMin * energyCapacity]
        constraints += [soc <= socMax * energyCapacity]
        constraints += [soc[0] == socInitial * energyCapacity]
        constraints += [soc[-1] == socFinal * energyCapacity]
        constraints += [soc[1:] == soc[:-1] + effCharge*powerInputCharge - effDischarge*powerInputDischarge] # added efficiency
        # Consumption
        powerConsumption = powerInput # positive consumption (cost added) when it charges, negative (cost avoided) when it discharges
        gasConsumption = np.zeros(n_timesteps)
        heatOutput = np.zeros(n_timesteps)
        # Cost
        capex = energyCapacity * capacityPrice # $
        CRF = discRate * (1 + discRate)**n_years / ((1 + discRate)**n_years - 1)

        super().__init__(name, typeTransfer, parameters, variables, variablesDict, constraints, powerConsumption, gasConsumption, heatOutput, capex, CRF)


    def describe(self):
        print(f"Component: {self.name}")
        if self._parameters is not None:
            for k, v in self._parameters.items():
                print(f"    {k}: {v}")
        print(f"    Optimal energy capacity: {np.round(self._variablesDict['energyCapacity'].value)} kWh")
        print(f"    Optimal power capacity: {np.round(self._parameters['maxChargeRate'] * self._variablesDict['energyCapacity'].value)} kW")


# Added power capacity to variables to optimize
class ThermalStorage(Component):

    def __init__(self, n_timesteps=None, dt=None, energyCapacityPrice=None, powerCapacityPrice=None,
                 lossRate=None, effCharge = None, effDischarge = None,
                 socMin=0, socMax=1, socInitial=0.5, socFinal=0.5, 
                 discRate=None, n_years=None):
        '''
        Inputs:
            - n_timesteps: number of time steps
            - dt: interval between time steps in hours
            - capacityPrice: price of capacity in $/kWh

            - lossRate: rate of energy loss per hour in % of storage capacity
            - effCharge: efficiency of the storage when charging
            - effDischarge: efficiency of the storage when discharging

            - socMin: minimum state of charge in % of storage capacity
            - socMax: maximum state of charge in % of storage capacity
            - socInitial: initial state of charge in % of storage capacity
            - socFinal: final state of charge in % of storage capacity
            
            - discRate: discount rate
            - n_years: lifetime of the component in years
        NB: State of charge soc is in kWh.
        '''
        # Defined default values directly in def init, as there is no maxChargeRate anymore
        # if maxChargeRate is not None:
        #     if maxDischargeRate is None:
        #         maxDischargeRate = maxChargeRate
        #     if socMin is None:
        #         socMin = 0
        #     if socMax is None:
        #         socMax = 1
        #     if socInitial is None:
        #         socInitial = 0.5
        #     if socFinal is None:
        #         socFinal = 0.5

        name = 'ThermalStorage'
        typeTransfer = 'Storage'
        # Parameters
        parameters = {'socMin': socMin, 'socMax': socMax, 'socInitial': socInitial, 'socFinal': socFinal,
                      'lossRate': lossRate, 'energyCapacityPrice': energyCapacityPrice, 'powerCapacityPrice': powerCapacityPrice,
                      'effCharge': effCharge, 'effDischarge': effDischarge}
        # Variables
        heatInputCharge = cp.Variable(n_timesteps) # kWh
        heatInputDischarge = cp.Variable(n_timesteps) # kWh
        heatInput = heatInputCharge - heatInputDischarge # kWh, positive when it charges, negative when it discharges
        soc = cp.Variable(n_timesteps + 1, nonneg=True) # kWh
        energyCapacity = cp.Variable(nonneg=True) # kWh
        powerCapacity = cp.Variable(nonneg=True) # kW
        variables = [heatInputCharge, heatInputDischarge, soc, energyCapacity]
        # Store variables in a dictionary
        variablesDict = {'heatInput': heatInput, 'heatInputCharge': heatInputCharge, 'heatInputDischarge': heatInputDischarge,
                         'soc': soc, 'energyCapacity': energyCapacity, 'powerCapacity': powerCapacity}
        # Constraints
        constraints = []
        constraints += [-heatInput <= powerCapacity * dt]
        constraints += [heatInput <= powerCapacity * dt]
        constraints += [soc >= socMin * energyCapacity]
        constraints += [soc <= socMax * energyCapacity]
        constraints += [soc[0] == socInitial * energyCapacity]
        constraints += [soc[-1] == socFinal * energyCapacity]
        constraints += [soc[1:] == soc[:-1]*(1 - dt*lossRate) + effCharge*heatInputCharge - effDischarge*heatInputDischarge] # added efficiency
        # Consumption
        powerConsumption = np.zeros(n_timesteps)
        gasConsumption = np.zeros(n_timesteps)
        heatOutput = - heatInput # load added when it charges (heatInput positive), load avoided when it discharges (heatInput negative)
        # Cost
        capex = energyCapacity * energyCapacityPrice + powerCapacity * powerCapacityPrice# $
        CRF = discRate * (1 + discRate)**n_years / ((1 + discRate)**n_years - 1)

        super().__init__(name, typeTransfer, parameters, variables, variablesDict, constraints, powerConsumption, gasConsumption, heatOutput, capex, CRF)

    
    def describe(self):
        print(f"Component: {self.name}")
        if self._parameters is not None:
            for k, v in self._parameters.items():
                print(f"    {k}: {v}")
        print(f"    Optimal energy capacity: {np.round(self._variablesDict['energyCapacity'].value)} kWhth")
        print(f"    Optimal heat capacity: {np.round(self._variablesDict['powerCapacity'].value)} kWth")


class PVsystem(Component):

    def __init__(self, n_timesteps=None, dt=None, pvLoadProfile=None, capacityPrice=None, ppaPrice=None,
                 discRate=None, n_years=None, onsite=False):
        '''
        Inputs:
            - n_timesteps: number of time steps
            - dt: interval between time steps in hours
            - pvLoadProfile: time-indexed electricity available from the PV system, in % of the PV capacity
            - capacityPrice: price of capacity in $/kW
            - ppaPrice: price of electricity obtained through PPA contracts in $/kWh
            - discRate: discount rate
            - n_years: lifetime of the component in years
            - onsite: boolean indicating whether the electricity is consumed on site or not
        '''
        # We suppose here that if the facility is not able to consume all the electricity produced by the PV system, the excess is put on the grid for free
        name = 'PVsystem'
        typeTransfer = 'ElectricityGeneration'
        # Variables
        capacity = cp.Variable(nonneg=True) # kW
        variables = [capacity]
        # Save variables in a dictionary    
        variablesDict = {'capacity': capacity}
        # Derived quantities
        powerOutput = pvLoadProfile * capacity * dt # kWh
        # Cost and parameter
        if capacityPrice is not None and ppaPrice is None:
            capex = capacity * capacityPrice # $
            CRF = discRate * (1 + discRate)**n_years / ((1 + discRate)**n_years - 1)
            parameters = {'capacityPrice': capacityPrice, 'pvLoadProfile': pvLoadProfile, 'onsite': onsite}
        elif ppaPrice is not None and capacityPrice is None:
            capex = cp.sum(powerOutput) * ppaPrice # In this case capex is already the annualized capex
            CRF = 1
            parameters = {'ppaPrice': ppaPrice, 'pvLoadProfile': pvLoadProfile, 'onsite': onsite}
        else:
            raise ValueError("Please provide either a capacity price or a levelized cost of electricity")
        # Constraints
        constraints = []
        # Consumption
        powerConsumption = - powerOutput
        gasConsumption = np.zeros(n_timesteps)
        heatOutput = np.zeros(n_timesteps)
        
        super().__init__(name, typeTransfer, parameters, variables, variablesDict, constraints, powerConsumption, gasConsumption, heatOutput, capex, CRF)
        
    def describe(self):
        print(f"Component: {self.name}")
        if self._parameters is not None:
            for k, v in self._parameters.items():
                print(f"    {k}: {v}")
        print(f"    Optimal power capacity: {np.round(self._variablesDict['capacity'].value/1000, 2)} MW")
    
class Windsystem(Component):

    def __init__(self, n_timesteps=None, dt=None, WindLoadProfile=None, capacityPrice=None, ppaPrice=None,
                 discRate=None, n_years=None, onsite=False):
        '''
        Inputs:
            - n_timesteps: number of time steps
            - dt: interval between time steps in hours
            - WindLoadProfile: time-indexed electricity available from the Wind system, in % of the Wind capacity
            - capacityPrice: price of capacity in $/kW
            - ppaPrice: price of electricity obtained through PPA contracts in $/kWh
            - discRate: discount rate
            - n_years: lifetime of the component in years
            - onsite: boolean indicating whether the electricity is consumed on site or not
        '''
        # We suppose here that if the facility is not able to consume all the electricity produced by the Wind system, the excess is put on the grid for free
        name = 'Windsystem'
        typeTransfer = 'ElectricityGeneration'
        # Variables
        capacity = cp.Variable(nonneg=True) # kW
        variables = [capacity]
        # Save variables in a dictionary    
        variablesDict = {'capacity': capacity}
        # Derived quantities
        powerOutput = WindLoadProfile * capacity * dt # kWh
        # Cost and parameter
        if capacityPrice is not None and ppaPrice is None:
            capex = capacity * capacityPrice # $
            CRF = discRate * (1 + discRate)**n_years / ((1 + discRate)**n_years - 1)
            parameters = {'capacityPrice': capacityPrice, 'WindLoadProfile': WindLoadProfile, 'onsite': onsite}
        elif ppaPrice is not None and capacityPrice is None:
            capex = cp.sum(powerOutput) * ppaPrice # In this case capex is already the annualized capex
            CRF = 1
            parameters = {'ppaPrice': ppaPrice, 'WindLoadProfile': WindLoadProfile, 'onsite': onsite}
        else:
            raise ValueError("Please provide either a capacity price or a levelized cost of electricity")
        # Constraints
        constraints = []
        # Consumption
        powerConsumption = - powerOutput
        gasConsumption = np.zeros(n_timesteps)
        heatOutput = np.zeros(n_timesteps)
        
        super().__init__(name, typeTransfer, parameters, variables, variablesDict, constraints, powerConsumption, gasConsumption, heatOutput, capex, CRF)
        
    def describe(self):
        print(f"Component: {self.name}")
        if self._parameters is not None:
            for k, v in self._parameters.items():
                print(f"    {k}: {v}")
        print(f"    Optimal power capacity: {np.round(self._variablesDict['capacity'].value/1000, 2)} MW")
