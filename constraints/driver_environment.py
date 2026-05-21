import random


class EnvironmentConstraints:

    def __init__(self):
        self.gps_std_dev    = 2.5   # Gaussian noise std dev in metres
        self.compliance_rate = 0.80  # 80% of drivers accept detour suggestions

    def apply_gps_noise(self, true_position):
        """Adds realistic Gaussian sensor noise to vehicle coordinates."""
        noisy_x = random.gauss(true_position[0], self.gps_std_dev)
        noisy_y = random.gauss(true_position[1], self.gps_std_dev)
        return (noisy_x, noisy_y)

    def check_driver_compliance(self, vehicle_id):
        """Returns True if the driver accepts the swarm detour suggestion."""
        return random.random() <= self.compliance_rate
