# TODO: Add sources
# TODO: find real vqlues

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
LION_EFF = 0.85 # []
LION_MAX_CHARGE_RATE = 1/4 # [1/h]
LION_CAPA_PRICE = 1587*LION_MAX_CHARGE_RATE # [$/kWhe]
LION_LIFETIME = 15 # [years]

PHYDRO_EFF = 0.8 # []
PHYDRO_MAX_CHARGE_RATE = 1/8 # [1/h]
PHYDRO_CAPA_PRICE = 2250*PHYDRO_MAX_CHARGE_RATE # [$/kWhe]
PHYDRO_LIFETIME = 15 # [years]

IRONAIR_EFF = 0.43 # []
IRONAIR_MAX_CHARGE_RATE = 1/100 # [1/h]
IRONAIR_CAPA_PRICE = 1400*IRONAIR_MAX_CHARGE_RATE # [$/kWhe]
IRONAIR_LIFETIME = 15 # [years]

H2_EFF = 0.34 # []
H2_MAX_CHARGE_RATE = 1/720 # [1/h]
H2_CAPA_PRICE = 2514*H2_MAX_CHARGE_RATE # [$/kWhe]
H2_LIFETIME = 15 # [years]

##### Thermal Storage
TES_EFF = 1 # [] # Already captured in the loss rate ?
TES_MAX_CHARGE_RATE = 1/100 # [1/h] # How do we know this ? Should we optimize for it ? I just came up with this value
TES_LOSS_RATE = 0.02 # [1/h]
TES_CAPA_PRICE = 1000*TES_MAX_CHARGE_RATE # [$/kWhth]
TES_LIFETIME = 20 # [years]