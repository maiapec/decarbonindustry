##### System
DISCOUNT_RATE = 0.055 # [] 

##### Natural gas Steam Boiler
# Source: https://energyinnovation.org/wp-content/uploads/2022/10/Decarbonizing-Low-Temperature-Industrial-Heat-In-The-U.S.-Report-2.pdf
FURNACE_EFF = 0.95 # [] 
FURNACE_CAPA_PRICE = 234 # [$/kW]
FURNACE_LIFETIME = 20 # [years]

##### Heat Pump
# Source: https://energyinnovation.org/wp-content/uploads/2022/10/Decarbonizing-Low-Temperature-Industrial-Heat-In-The-U.S.-Report-2.pdf
HP_COP = 3.7 # [] Value corresponding to low temperature processes (80-100°C). Typical values range between 3 and 5.
HP_CAPA_PRICE = 700 # [$/kW]
HP_LIFETIME = 15 # [years]

##### Battery

# Source: Li-ion LFP 2023 estimates from https://www.pnnl.gov/ESGC-cost-performance
LION_EFF_CHARGE = 0.83 # []
LION_EFF_DISCHARGE = 0.83 # []
LION_MAX_CHARGE_RATE = 1/4 # [1/h] (Duration of storage: 4 hours)
LION_CAPA_PRICE = 405 # [$/kWhe]
LION_LIFETIME = 16 # [years]

# Source: Lead-acid 2023 estimates from https://www.pnnl.gov/ESGC-cost-performance
LEADACID_EFF_CHARGE = 0.77 # []
LEADACID_EFF_DISCHARGE = 0.77 # []
LEADACID_MAX_CHARGE_RATE = 1/4 # [1/h] (Duration of storage: 4 hours)
LEADACID_CAPA_PRICE = 458 # [$/kWhe]
LEADACID_LIFETIME = 14 # [years]

# Source: Pumped hydro storage 2023 estimates from https://www.pnnl.gov/ESGC-cost-performance
PHYDRO_EFF_CHARGE = 0.8 # []
PHYDRO_EFF_DISCHARGE = 0.8 # []
PHYDRO_MAX_CHARGE_RATE = 1/10 # [1/h]
PHYDRO_CAPA_PRICE = 279 # [$/kWhe]
PHYDRO_LIFETIME = 60 # [years]

# Source: Hydrogen bi-directional fuel-cell (BDFC) storage 2023 estimates from https://www.pnnl.gov/ESGC-cost-performance
H2BDFC_EFF_CHARGE = 0.31 # []
H2BDFC_EFF_DISCHARGE = 0.31 # []
H2BDFC_MAX_CHARGE_RATE = 1/24 # [1/h]
H2BDFC_CAPA_PRICE = 126 # [$/kWhe]
H2BDFC_LIFETIME = 30 # [years]

# Iron air?
# IRONAIR_EFF_CHARGE = 0.43 # []
# IRONAIR_EFF_DISCHARGE = 0.43 # []
# IRONAIR_MAX_CHARGE_RATE = 1/100 # [1/h]
# IRONAIR_CAPA_PRICE = 1400*IRONAIR_MAX_CHARGE_RATE # [$/kWhe]
# IRONAIR_LIFETIME = 15 # [years]


##### Thermal Storage
# Source: 10-hour thermal storage 2023 estimates from https://www.pnnl.gov/ESGC-cost-performance
# NB: 24-hour storage has ECAPA price of 101 $/kWhth and PCAPA price of 1627 $/kWth.
TES_EFF_CHARGE = 0.5 # [] 
TES_EFF_DISCHARGE = 0.5 # []
TES_LOSS_RATE = 0.001 # [1/h] TODO: Find this value.
TES_ECAPA_PRICE = 166 # [$/kWhth] # Previous values from Aramis was 10
TES_PCAPA_PRICE = 1466 # [$/kWth] # Previous values from Aramis was 50
TES_LIFETIME = 34 # [years]

##### Solar
PV_AVG_EMISSIONS = 0.0553 # [kgCO2/kWh] # Mean life-cycle emissions for c-Si PV.
                          # Source: https://doi.org/10.1016/j.enpol.2013.10.048
PV_OFFSITE_PPA_PRICE = 0.04 # [$/kWh] LCOE for class 5 (default) utility-scale PV in 2022. This value is used as the PPA price.
                               # Source: https://atb.nrel.gov/electricity/2023/technologies
PV_ONSITE_PCAPA_PRICE = 1900 # [$/kW] Power capacity price for class 5 (default) commercial, mature PV technology in 2022.
                                # Source: https://atb.nrel.gov/electricity/2023/technologies
PV_ONSITE_LIFETIME = 25 # [years] or 30

##### Wind
WIND_AVG_EMISSIONS = 0.015 # [kgCO2/kWh] # Mean life-cycle emissions for onshore wind.
                           # Source: https://doi.org/10.1111/j.1530-9290.2012.00464.x
WIND_OFFSITE_PPA_PRICE = 0.03 # [$/kWh] LCOE for class 4 (default) onshore PV in 2022. This value is used as the PPA price.
                                 # Source: https://atb.nrel.gov/electricity/2023/technologies
WIND_ONSITE_PCAPA_PRICE = 1550 # [$/kW] Power capacity price for class 4 (default) onshore PV technology in 2022.
                                  # Source: https://atb.nrel.gov/electricity/2023/technologies
WIND_ONSITE_LIFETIME = 20 # [years] or 30