#Nuevo archivo
from config_builder import ConfigBuilder
import json, os

def main():
    builder = ConfigBuilder()
    config = builder.build_config()

    # Guardar la configuración en config.json
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"\nArchivo config.json generado exitosamente en {config_path}.")

if __name__ == "__main__":
    main()