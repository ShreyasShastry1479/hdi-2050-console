import pandas as pd
import wbgapi as wb

# Try more education indicators
for ind in ["SE.XPD.TERTH.PC.SS", "SE.PRM.ENRR", "SE.SEC.ENRR.TC", "SE.TER.ENRR.TC", "SE.ADT.1524.LT.ZS"]:
    try:
        data = wb.data.DataFrame(ind, ["USA","ESP","NGA","IND"], time=range(2020,2025), columns="time", numericTimeKeys=True)
        print(f"\n{ind}:")
        print(data)
    except Exception as e:
        print(f"{ind}: error")
