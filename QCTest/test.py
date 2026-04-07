import sys
import pandas as pd

sys.path.append("/var/www/python/Qingcheng/nighthawk/")
from nighthawk.util import sql_functions
from nighthawk.data.pipeline.common_functions.load import Load
from nighthawk.data.pipeline.common_functions.wind import Wind
from nighthawk.data.pipeline.common_functions.solar import Solar
from nighthawk.data.pipeline.common_functions.genoutage import GenOutage
from nighthawk.data.pipeline.var_handler.fuel_type import FuelType

START = "2026-04-01"
END   = "2026-04-01"
BAA_ZONES = ["E", "W"]

load     = Load("SPP")
wind     = Wind("SPP")
solar    = Solar("SPP")
genout   = GenOutage("SPP")
fueltype = FuelType("SPP")

from nighthawk.data.pipeline.var_handler.fuel_type import get_data_and_mapping_for_genoutage_fuel_type,get_data_and_mapping_for_gen_fuel_type
data, mapping = get_data_and_mapping_for_gen_fuel_type(['636'], 'SPP', '2026-03-30', '2026-03-30', var_spec=['f', 'a'])

print(data)
ft = FuelType('SPP')
D = '2026-04-01'

# ── 1. get_gen_fuel_type ─────────────────────────────────────────────────────
# GenMixHourly → Hydro, Nuclear, NaturalGas only; forecast = 1-day shift of actual
df1 = ft.get_gen_fuel_actual(D, '2026-04-02') 
print('1. get_gen_fuel_type:', df1.shape)
print('   columns:', df1.columns.tolist())
print(df1)