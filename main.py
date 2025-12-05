#-----------------------------------------------------------------------------#
# ---- NATIVE MODULES
#-----------------------------------------------------------------------------#
import os
import importlib

#-----------------------------------------------------------------------------#
# ---- NON-NATIVE MODULES
#-----------------------------------------------------------------------------#
from dask.distributed import LocalCluster, Client

#-----------------------------------------------------------------------------#
# ---- DISTRIBUTED ENVIRONMENT
#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    client = Client(LocalCluster())

#-----------------------------------------------------------------------------#
# ---- DATA MODULES
#-----------------------------------------------------------------------------#
folder_fs = "data"
folder_pkg = folder_fs.replace("/", ".")
for file in os.listdir(folder_fs):
    if file.endswith(".py") and file != "__init__.py":
        module_name = os.path.splitext(file)[0]
        module = importlib.import_module(f"{folder_pkg}.{module_name}")
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if callable(attr):  # Check if it's a callable (function, method, etc.)
                globals()[attr_name] = attr  # Add it to the global namespace

#-----------------------------------------------------------------------------#
# ---- PROBABILITY DISTRIBUTION MODULES
#-----------------------------------------------------------------------------#
# Continuous Distributions
folder_fs = "probability_distributions/continuous"
folder_pkg = folder_fs.replace("/", ".")
for file in os.listdir(folder_fs):
    if file.endswith(".py") and file != "__init__.py":
        module_name = os.path.splitext(file)[0]
        module = importlib.import_module(f"{folder_pkg}.{module_name}")
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if callable(attr):  # Check if it's a callable (function, method, etc.)
                globals()[attr_name] = attr  # Add it to the global namespace

# Discrete Distributions
folder_fs = "probability_distributions/discrete"
folder_pkg = folder_fs.replace("/", ".")
for file in os.listdir(folder_fs):
    if file.endswith(".py") and file != "__init__.py":
        module_name = os.path.splitext(file)[0]
        module = importlib.import_module(f"{folder_pkg}.{module_name}")
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if callable(attr):  # Check if it's a callable (function, method, etc.)
                globals()[attr_name] = attr  # Add it to the global namespace

#-----------------------------------------------------------------------------#
# ---- REGRESSION MODULES
#-----------------------------------------------------------------------------#
folder_fs = "regression"
folder_pkg = folder_fs.replace("/", ".")
for file in os.listdir(folder_fs):
    if file.endswith(".py") and file != "__init__.py":
        module_name = os.path.splitext(file)[0]
        module = importlib.import_module(f"{folder_pkg}.{module_name}")
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if callable(attr):  # Check if it's a callable (function, method, etc.)
                globals()[attr_name] = attr  # Add it to the global namespace

#-----------------------------------------------------------------------------#
# ---- CLEAN ENVIRONMENT
#-----------------------------------------------------------------------------#
del attr_name, file, folder_fs, folder_pkg, module_name
