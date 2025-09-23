import numpy as np
import json
import pandas as pd # Añadir pandas para manejar Timestamp si es necesario

class NumpyEncoder(json.JSONEncoder):
    """
    Codificador JSON personalizado para manejar tipos de NumPy (ndarray, float, int)
    y potencialmente otros tipos como Timestamp de Pandas.
    """
    def default(self, obj):
        # Manejar arrays de NumPy
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Manejar tipos flotantes de NumPy
        if isinstance(obj, (np.float16, np.float32, np.float64)):
            # Convertir NaN/inf a strings o None para JSON válido
            if pd.isna(obj) or not np.isfinite(obj):
                return None # o 'NaN', 'Infinity', '-Infinity' si se prefiere string
            return float(obj)
        # Manejar tipos enteros de NumPy
        if isinstance(obj, ((np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64))):
            return int(obj)
        # Manejar booleanos de NumPy
        if isinstance(obj, np.bool_):
             return bool(obj)
        # Manejar Timestamps de Pandas (si se usan en los datos)
        if isinstance(obj, pd.Timestamp):
             return obj.isoformat() # Convertir a string ISO 8601

        # Dejar que el codificador base maneje otros tipos o lance error
        return super(NumpyEncoder, self).default(obj)