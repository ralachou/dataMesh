# Run this in a notebook cell (same env)
from notebook.services.config import ConfigManager
from pathlib import Path
import json, sys

cm = ConfigManager()
cm.update('notebook', {"load_extensions": {"widgetsnbextension/extension": True}})

# Also write explicit config files (belt & suspenders)
base = Path(sys.prefix) / "etc" / "jupyter" / "nbconfig"
home = Path.home() / ".jupyter" / "nbconfig"
for d in (base, home):
    d.mkdir(parents=True, exist_ok=True)
    p = d / "notebook.json"
    try:
        cfg = json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        cfg = {}
    cfg.setdefault("load_extensions", {})
    cfg["load_extensions"]["widgetsnbextension/extension"] = True
    p.write_text(json.dumps(cfg, indent=2))

print("Widgets enabled. Now restart the notebook server and hard-refresh the browser.")
