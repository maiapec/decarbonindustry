# Data is adapted from the NREL 2023 Annual Technology Baseline.
# https://atb.nrel.gov/electricity/2023/technologies

##### System
DISCOUNT_RATE = 0.05 # [] 

##### Gas Furnace
FURNACE_EFF = 0.85 # []
FURNACE_CAPA_PRICE = 200 # [$/kW]
FURNACE_LIFETIME = 20 # [years]

##### Heat Pump
HP_COP = 3 # []
HP_CAPA_PRICE = 1000 # [$/kW]
HP_LIFETIME = 15 # [years]

##### Battery
LION_EFF_CHARGE = 0.85 # []
LION_EFF_DISCHARGE = 0.85 # []
LION_MAX_CHARGE_RATE = 1/4 # [1/h]
LION_CAPA_PRICE = 1587*LION_MAX_CHARGE_RATE # [$/kWhe]
LION_LIFETIME = 15 # [years]

PHYDRO_EFF_CHARGE = 0.8 # []
PHYDRO_EFF_DISCHARGE = 0.8 # []
PHYDRO_MAX_CHARGE_RATE = 1/8 # [1/h]
PHYDRO_CAPA_PRICE = 2250*PHYDRO_MAX_CHARGE_RATE # [$/kWhe]
PHYDRO_LIFETIME = 15 # [years]

IRONAIR_EFF_CHARGE = 0.43 # []
IRONAIR_EFF_DISCHARGE = 0.43 # []
IRONAIR_MAX_CHARGE_RATE = 1/100 # [1/h]
IRONAIR_CAPA_PRICE = 1400*IRONAIR_MAX_CHARGE_RATE # [$/kWhe]
IRONAIR_LIFETIME = 15 # [years]

H2_EFF_CHARGE = 0.34 # []
H2_EFF_DISCHARGE = 0.34 # []
H2_MAX_CHARGE_RATE = 1/720 # [1/h]
H2_CAPA_PRICE = 2514*H2_MAX_CHARGE_RATE # [$/kWhe]
H2_LIFETIME = 15 # [years]

##### Thermal Storage
TES_EFF_CHARGE = 1 # [] 
TES_EFF_DISCHARGE = 1 # []
#TES_MAX_CHARGE_RATE = 1/100 # [1/h] # not used now because defined a variable for max power
TES_LOSS_RATE = 0 # [1/h]
TES_CAPA_PRICE = 200 # [$/kWhth]
TES_LIFETIME = 20 # [years]

##### Solar PPA
PV_PPA_PRICE = 0.04 # [$/kWh]
PV_AVG_EMISSIONS = 0.03 # [kgCO2/kWh] # I came up with this value
