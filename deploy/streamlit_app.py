"""Punto de entrada para Streamlit Community Cloud.

Ejecuta el mismo ``app.py`` de la raíz. Existe para que Streamlit Cloud tome las
dependencias de ``deploy/requirements.txt`` (pip): la instalación con conda del
``environment.yml`` no termina en la máquina gratuita de Streamlit. En local se
sigue usando conda y ``streamlit run app.py``.
"""

import runpy
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

runpy.run_path(str(RAIZ / "app.py"), run_name="__main__")
