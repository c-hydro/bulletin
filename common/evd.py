import numpy as np
import logging

class Distribution:
    def calculate_return_period(self, Q: np.ndarray) -> np.ndarray:
        """
        Calculate the return period.

        :param Q: Array of discharge values.
        :return: Array of return periods.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

class GEVDistribution(Distribution):
    def __init__(self, theta1: np.ndarray, theta2: np.ndarray, theta3: np.ndarray):
        """
        Initialize the GEVDistribution.

        :param theta1: Array of theta1 values.
        :param theta2: Array of theta2 values.
        :param theta3: Array of theta3 values.
        """
        logging.info("Initializing GEV Distribution with parameters")
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3

    @staticmethod
    def from_maps(theta1_map: np.ndarray, theta2_map: np.ndarray, theta3_map: np.ndarray) -> 'GEVDistribution':
        """
        Create a GEVDistribution from maps.

        :param theta1_map: Array of theta1 values.
        :param theta2_map: Array of theta2 values.
        :param theta3_map: Array of theta3 values.
        :return: GEVDistribution object.
        """
        logging.info("Creating GEV Distribution from maps")
        return GEVDistribution(theta1_map, theta2_map, theta3_map)

    def calculate_return_period(self, Q: np.ndarray) -> np.ndarray:
        """
        Calculate the return period.

        :param Q: Array of discharge values.
        :return: Array of return periods.
        """
        value = 1 - self.theta3 * ((Q - self.theta1) / self.theta2)
        value[value < 0] = 0
        y = np.where(self.theta3 == 0, (Q - self.theta1) / self.theta2, (-1 / self.theta3) * np.log(value))
        P = np.exp(-np.exp(-y))
        T = np.floor(1 / (1 - P))
        return T

def get_distribution(distribution_name: str, params_maps: dict[str, np.ndarray]) -> Distribution:
    """
    Get a distribution object.

    :param distribution_name: Name of the distribution.
    :param params_maps: Dictionary of parameter maps.
    :return: Distribution object.
    """
    logging.info(f"Getting distribution: {distribution_name}")
    if distribution_name == "GEV":
        return GEVDistribution.from_maps(params_maps["theta1"], params_maps["theta2"], params_maps["theta3"])
    else:
        logging.error(f"Unsupported distribution: {distribution_name}")
        raise ValueError(f"Unsupported distribution: {distribution_name}")