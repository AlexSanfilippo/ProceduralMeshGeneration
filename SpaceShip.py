"""
Holds everything related to the ships'
    -movement
        -position
        -velocity
        -acceleration
        -direction vector
    -status
        -fuel
        -damage

"""
import glm
class Spaceship():
    """

    """

    def __init__(self, model):
        self.model = model
        self.velocity = glm.vec3(0.0, 0.0, 0.0)
        self.fuel_current = 100.0 #space gallons
        self.fuel_capacity = 100.0
        self.engine_efficiency = 1.0 #space gallons per hour=1.0 delta_time
        self.target = None

    def draw(self, view):
        self.model.draw(view=view)

    def move(self, delta_time=0.0):
        if self.fuel_current < 0.0000001:
            return
        self.model.position += self.velocity*delta_time
        self.fuel_current -= delta_time*self.engine_efficiency

    def at_target(self):
        """
        check if ship is close enough to target to interact (trade, land, etc)
        :return: boolean
        """
        return False
    def set_velocity(self, velocity=[0.0, 0.0, 0.0]):
        self.velocity = glm.vec3(velocity)

    def set_target(self, target):
        """
        Give the ship a target (like a planet object) to move towards
        :param target: A class like planet. Must have a position
        :return:
        """
        self.target = target

