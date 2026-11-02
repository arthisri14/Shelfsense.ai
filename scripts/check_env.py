import sys
import numpy, pandas, scipy, statsmodels, sklearn, pydantic, fastapi, jwt, streamlit

print("Python:", sys.version.split()[0])
for m in (numpy, pandas, scipy, statsmodels, sklearn, pydantic, fastapi, jwt, streamlit):
    print(f"{m.__name__:12} {m.__version__}")
print("Environment OK")