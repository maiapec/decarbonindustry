from pathlib import Path
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from energySystem import System, NaturalGasBoiler, HeatPump, Battery, ThermalStorage, PVsystem, Windsystem, ElectricBoiler
import layouts.default_values as DEFAULT

sns.set_context(context="notebook")
plt.rcParams["figure.dpi"] = 300
path = Path('.') / "layouts"
colors = {'HeatPump': 'C0', 'Power Load': 'C7', 'Lithium Ion Battery': 'C4', 'Total Power Consumption': 'black', 'PVsystem': 'C1', 'Windsystem':'C2', '':'C4',
          'NaturalGasBoiler' : 'C5', 'ElectricBoiler': 'C6', 'Heat Load': 'C3', 'ThermalStorage': 'C4', 'Total Heat Consumption': 'black',
          'Marginal Power Price': 'C3', 'Marginal Power Emissions': 'C5'}

def get_demand_mask_and_price(df, colName, dt):
    col = df[colName].to_numpy() / dt
    mask = col > 0
    price = col[mask][0]
    return mask, price

def load_site_data(sites):
    if not isinstance(sites, list):
        sites = [sites]
    powerLoad, heatLoad = 0, 0
    for site in sites:
        df = pd.read_csv(path / "loads" / (site + "_load.csv"))
        powerLoad = powerLoad + df['Power Load [kWh]'].to_numpy()
        heatLoad = heatLoad + df['Heat Load [kWh]'].to_numpy()
    return powerLoad, heatLoad

def load_prices(directAccess, dt):
    if directAccess:
        df = pd.read_csv(path / "power_grid" / "direct_access.csv")
    df = pd.read_csv(path / "power_grid" / "power_price_B20.csv")
    energyPricePower = df["energyPrice"].to_numpy() # $/kWh
    sellBackPrice = np.full(len(energyPricePower), DEFAULT.SELL_BACK_PRICE) # $/kWh
    powerDemandFee = [
        get_demand_mask_and_price(df, colName, dt) for colName in [
            "peakDemandSummerPrice",
            "partialPeakDemandSummerPrice",
            "demandSummerPrice",
            "peakDemandWinterPrice",
            "demandWinterPrice"
            ]]
    df = pd.read_csv(path / "gas" / "gas_price.csv")
    energyPriceGas = df["energyPrice"].to_numpy() # $/kWh
    return energyPricePower, sellBackPrice, powerDemandFee, energyPriceGas

def load_emissions():
    df = pd.read_csv(path / "power_grid" / "power_grid_emissions.csv")
    df.ffill(inplace=True)
    emissionsPower = df["MOER version 2.0"].to_numpy() # kCO2eq/KWhe
    df = pd.read_csv(path / "gas" / "gas_emissions.csv")
    emissionsGas = df["gasEmissions"].to_numpy() # kCO2eq/kWhgas
    return emissionsPower, emissionsGas

def load_renewable_profiles():
    # pv load
    df = pd.read_csv(path / "renewable_loadprofiles" / "utility_scale_solar.csv")
    pvuLoad = df["0"].to_numpy()
    # wind load
    df = pd.read_csv(path / "renewable_loadprofiles" / "wind.csv")
    windLoad = df["feedin_power_plant"].to_numpy()
    return pvuLoad, windLoad

def load_data(directAccess, sites):
    dt = 1/4 # in hours
    powerLoad, heatLoad = load_site_data(sites) # loads
    energyPricePower, sellBackPrice, powerDemandFee, energyPriceGas = load_prices(directAccess, dt) # prices
    emissionsPower, emissionsGas = load_emissions() # emissions
    pvuLoad, windLoad = load_renewable_profiles() # renewable profiles
    timeIndex = pd.date_range(start='1/1/2023', periods=len(powerLoad), freq='15min') # time index
    return timeIndex, powerLoad, heatLoad, energyPricePower, sellBackPrice, powerDemandFee, energyPriceGas, emissionsPower, emissionsGas, pvuLoad, windLoad, dt

def build_gas_only_system(name, directAccess=False, site="site1"):
    timeIndex, powerLoad, heatLoad, energyPricePower, sellBackPrice, powerDemandFee, energyPriceGas, emissionsPower, emissionsGas, pvuLoad, windLoad, dt = load_data(directAccess, site)
    n_timesteps = len(timeIndex)
    system = System(
        name,
        timeIndex=timeIndex,
        powerLoad=powerLoad,
        heatLoad=heatLoad,
        powerPrice=energyPricePower,
        sellBackPrice=sellBackPrice,
        powerDemandFee=powerDemandFee,
        gasPrice=energyPriceGas,
        gridMarginalEmissions=emissionsPower,
        gasMarginalEmissions=emissionsGas
    )
    system.addComponent(NaturalGasBoiler(
        n_timesteps=n_timesteps,
        dt=1/4,
        eff=DEFAULT.GAS_BOILER_EFF,
        capacityPrice=DEFAULT.GAS_BOILER_CAPA_PRICE,
        discRate=DEFAULT.DISCOUNT_RATE,
        n_years=DEFAULT.GAS_BOILER_LIFETIME
        ))
    return system

def build_complete_system(name, directAccess=False, site="site1"):
    timeIndex, powerLoad, heatLoad, energyPricePower, sellBackPrice, powerDemandFee, energyPriceGas, emissionsPower, emissionsGas, pvuLoad, windLoad, dt = load_data(directAccess, site)
    n_timesteps = len(timeIndex)
    system = System(
        name,
        timeIndex=timeIndex,
        powerLoad=powerLoad,
        heatLoad=heatLoad,
        powerPrice=energyPricePower,
        sellBackPrice=sellBackPrice,
        powerDemandFee=powerDemandFee,
        gasPrice=energyPriceGas,
        gridMarginalEmissions=emissionsPower,
        gasMarginalEmissions=emissionsGas
    )
    system.addComponent(NaturalGasBoiler(
        n_timesteps=n_timesteps,
        dt=1/4,
        eff=DEFAULT.GAS_BOILER_EFF,
        capacityPrice=DEFAULT.GAS_BOILER_CAPA_PRICE,
        discRate=DEFAULT.DISCOUNT_RATE,
        n_years=DEFAULT.GAS_BOILER_LIFETIME
        ))
    system.addComponent(HeatPump(
        n_timesteps=n_timesteps,
        dt=1/4,
        COP=DEFAULT.HP_COP,
        ramp_rate=DEFAULT.HP_RAMP_RATE,
        capacityPrice=DEFAULT.HP_CAPA_PRICE,
        discRate=DEFAULT.DISCOUNT_RATE,
        n_years=DEFAULT.HP_LIFETIME
        ))
    system.addComponent(Battery(
        n_timesteps=n_timesteps,
        dt=1/4,
        capacityPrice=DEFAULT.LION_CAPA_PRICE,
        maxChargeRate=DEFAULT.LION_MAX_CHARGE_RATE,
        effCharge=DEFAULT.LION_EFF_CHARGE,
        effDischarge=DEFAULT.LION_EFF_DISCHARGE,
        discRate=DEFAULT.DISCOUNT_RATE,
        n_years=DEFAULT.LION_LIFETIME,
        name="Lithium Ion Battery"
        ))
    system.addComponent(ThermalStorage(
        n_timesteps=n_timesteps,
        dt=1/4,
        energyCapacityPrice=DEFAULT.TES_ECAPA_PRICE,
        powerCapacityPrice=DEFAULT.TES_PCAPA_PRICE,
        lossRate=DEFAULT.TES_LOSS_RATE,
        effCharge=DEFAULT.TES_EFF_CHARGE,
        effDischarge=DEFAULT.TES_EFF_DISCHARGE,
        discRate=DEFAULT.DISCOUNT_RATE,
        n_years=DEFAULT.TES_LIFETIME
        ))
    system.addComponent(PVsystem(
        n_timesteps=n_timesteps,
        dt=1/4,
        pvLoadProfile=pvuLoad,
        ppaPrice=DEFAULT.PV_OFFSITE_PPA_PRICE,
        onsite=False
        ))
    system.addComponent(Windsystem(
        n_timesteps=n_timesteps,
        dt=1/4,
        WindLoadProfile=windLoad,
        ppaPrice=DEFAULT.WIND_OFFSITE_PPA_PRICE,
        onsite=False
        ))
    system.addComponent(ElectricBoiler(
        n_timesteps=n_timesteps,
        dt=1/4,
        eff=DEFAULT.ELECTRIC_BOILER_EFF,
        capacityPrice=DEFAULT.ELECTRIC_BOILER_CAPA_PRICE,
        discRate=DEFAULT.DISCOUNT_RATE,
        n_years=DEFAULT.ELECTRIC_BOILER_LIFETIME
        ))
    return system

def save_results(system, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    system.saveResults(directory)

def run_scenario(system, scenario):
    if scenario == "MinimizeCost":
        system.solve(objective="cost", solver="MOSEK")
    directory = path / "results" / (system.name + "_" + scenario)
    save_results(system, directory)
    return system._status

def load_power_and_heat(systemName, obj):
    directory = path / "results" / (systemName + "_" + obj)
    power = pd.read_csv(directory / "power.csv", index_col=0, parse_dates=True)
    heat = pd.read_csv(directory / "heat.csv", index_col=0, parse_dates=True)
    return power, heat

def plot_power(pwr, powerConsumers, powerStorage, powerGenerators, period=None, start=None, end=None):
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
    
def plot_heat(power, heat, heatGenerators, heatStorage, period=None, start=None, end=None):
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

def plot_power_and_heat(power, heat, powerConsumers, powerStorage, powerGenerators, heatGenerators, heatStorage, period=None, start=None, end=None):
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
    power_pos = power[powerConsumers].copy() 
    power_neg = power[powerGenerators].copy()
    power_batt_pos, power_battery_neg = power[powerStorage].clip(lower=0), power[powerStorage].clip(upper=0)
    power_pos[powerStorage] = power_batt_pos
    power_neg[''] = power_battery_neg
    heat_pos = heat[heatGenerators].copy()
    heat_sto_pos, heat_sto_neg = heat[heatStorage].clip(lower=0), heat[heatStorage].clip(upper=0)
    heat_pos[heatStorage] = heat_sto_pos
    # Plot
    fig, axs = plt.subplots(2, figsize=(15, 10), dpi=300, sharex=True)
    power_pos.plot.area(color=colors, ax=axs[0])
    power_neg.plot.area(color=colors, ax=axs[0])
    power['Total Power Consumption'].plot(color=colors, ax=axs[0])
    axs[0].set_ylabel('Power Consumption(kW)')
    axs[0].legend()
    heat_pos.plot.area(color=colors, ax=axs[1])
    for heatSto in heatStorage:
        heat_sto_neg[heatSto].plot.area(color=colors[heatSto], label='', ax=axs[1])
    heat['Heat Load'].plot(color=colors['Heat Load'], ax=axs[1])
    axs[1].set_ylabel('Heat Generation(kW)')
    axs[1].legend()
    plt.tight_layout
    return plt.gca()
