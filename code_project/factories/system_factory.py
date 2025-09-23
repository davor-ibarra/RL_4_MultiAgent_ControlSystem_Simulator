from components.systems.inverted_pendulum_system import InvertedPendulumSystem

class SystemFactory:
    @staticmethod
    def create_system(system_config):
        system_type = system_config.get('type')
        
        if system_type == 'inverted_pendulum':
            return InvertedPendulumSystem(**system_config['params'])
        
        raise ValueError(f"System '{system_type}' not recognized.")
