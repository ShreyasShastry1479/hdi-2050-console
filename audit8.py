import pandas as pd
import wbgapi as wb

# Check what SE.ADT.LITR.ZS actually is
try:
    desc = wb.indicator.metadata("SE.ADT.LITR.ZS")
    print("SE.ADT.LITR.ZS:", desc.get('value','')[:200])
except:
    pass

# Check actual values for a few countries
try:
    data = wb.data.DataFrame("SE.ADT.LITR.ZS", ["USA","ESP","NGA","IND"], time=range(2020,2025), columns="time", numericTimeKeys=True)
    print("\nActual literacy rates (SE.ADT.LITR.ZS):")
    print(data)
except Exception as e:
    print(f"Error: {e}")

# Better indicators for HDI education
for ind in ["SE.XPD.TERTH.PC.SS", "SE.PRM.ENRR.TC", "SE.SEC.ENRR", "SE.TER.ENRR"]:
    try:
        data = wb.data.DataFrame(ind, ["USA","ESP","NGA","IND"], time=range(2020,2025), columns="time", numericTimeKeys=True)
        print(f"\n{ind}:")
        print(data)
    except Exception as e:
        print(f"{ind}: error {e}")
