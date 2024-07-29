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
import threading
from pydub.playback import play

import glm
from math import pi, floor, asin
from collections import OrderedDict, defaultdict

from pydub import AudioSegment

from ProceduralMesh import Spaceship as SpaceshipModel

sound = AudioSegment.from_wav("Sounds/ShipDeposit.wav")


class Spaceship:
    """

    """
    def __init__(self, model, target=None, wallet=0):
        self.model = model
        self.velocity = glm.vec3(0.0, 0.0, 0.0)
        self.fuel_current = 100.0 #space gallons
        self.fuel_capacity = 100.0
        self.engine_efficiency = 1.0 #space gallons per hour=1.0 delta_time
        self.target = None
        self.state = None
        self.orders = ['move', target]
        self.index_orders = 0
        self.cargo = defaultdict(int)
        self.wallet = wallet
        self.data = {}
        self.minimum_funds = 50
        self.max_cargo_weight = 100
        self.name = 'enterprise'
        self.profit = 0


    def set_name(self, name):
        self.name = name
    def set_cargo(self, cargo):
        self.cargo = defaultdict(int)
        for good, quantity in cargo.items():
            self.cargo[good] = quantity

    def set_orders(self, orders, index=0):
        if not orders:
            return
        self.orders = orders
        self.index = index
        self.state = self.orders[index][0]
        if self.state == 'move':
            self.set_target(self.orders[self.index_orders][1])

    def draw(self, view):
        self.model.draw(view=view)

    def order_move(self, delta_time=0.0, use_fuel=False):
        if not self.target:
            self.state = 'stuck'
            return
        if self.fuel_current < 0.0000001 and use_fuel:
            self.state = 'stuck'
            return
        self.rotate_to_target()
        self.move(delta_time)
        if use_fuel:
            self.fuel_current -= delta_time*self.engine_efficiency
        if self.at_target():
            if type(self.target) is Headquarters:
                self.upload_data(headquarters=self.target)
                self.deposit_funds(headquarters=self.target)
                self.goto_next_order()
            else:
                self.download_data()
                self.goto_next_order()
    def upload_data(self, headquarters):
        if not headquarters:
            print("ERROR: cannot upload data, headquarters is None")
            return
        for planet in self.data.keys():
            headquarters.data[planet] = self.data[planet]

    def download_data(self):
        """
        download data from target to ship computer banks
        :return: No Return
        """
        self.data[self.target.name] = self.target.prices

    def deposit_funds(self, headquarters):
        deposit = max(0, self.wallet - self.minimum_funds)
        headquarters.wallet += deposit
        self.wallet -= deposit
        self.profit = deposit

    def move(self, delta_time):
        self.model.position += self.velocity * delta_time

    def process_states(self, delta_time=0.0001):
        if self.state == 'move':
            self.order_move(delta_time=delta_time, use_fuel=False)
        elif self.state == 'merchant':
            self.order_sell()
            self.order_buy()
            self.goto_next_order()
        elif self.state == 'park':
            pass


    def goto_next_order(self):
        """
        move to next value in order
            -is order a list or an ordered dictionary?

        :return:
        """
        self.index_orders += 1
        if self.index_orders >= len(self.orders):
            self.index_orders = 0
        self.state = self.orders[self.index_orders][0]
        if self.state == 'move':
            self.target = self.orders[self.index_orders][1]



    def order_sell(self):
        good = self.orders[self.index_orders][1]['sell']
        if self.cargo[good] > 0 and self.target.prices[good] > 0:
            self.wallet += self.target.prices[good] * self.cargo[good]
            print('sold ', self.cargo[good], ' at $', self.target.prices[good], ' for ', self.target.prices[good] * self.cargo[good])
            self.cargo[good] = 0

        # if my_thread.is_alive():
        #     my_thread.run()
        # else:
        my_thread = threading.Thread(
            target=play,
            kwargs={'audio_segment': sound},
            daemon=True,
        )
        my_thread.start()

        # main program waits for thread to finish
        # my_thread.join()



    def order_buy(self):
        good = self.orders[self.index_orders][1]['buy']
        if self.target.cargo[good] > 0:
            price = self.target.prices[good]
            quantity_available = self.target.cargo[good]
            quantity_to_buy = min(floor(self.wallet / price), quantity_available)
            self.wallet -= quantity_to_buy*price
            self.target.cargo[good] -= quantity_to_buy
            self.cargo[good] += quantity_to_buy
            print('bought ', quantity_to_buy, ' tons of ', good, ' at $', price)

    def at_target(self):
        """
        check if ship is close enough to target to interact (trade, land, etc)
        :return: boolean
        """
        if glm.distance(self.model.position, self.target.position) < self.target.trade_distance:
            return True
        return False
    def set_velocity(self, velocity=(0.0, 0.0, 0.0)):
        self.velocity = glm.vec3(velocity)

    def set_target(self, target):
        """
        Give the ship a target (like a planet object) to move towards
        :param target: A class like planet. Must have a position
        :return:
        """
        self.target = target
        self.state = 'move'

    def rotate_to_target(self):
        """
        Rotate the model to face the position of the target
        :return: No return
        """
        direction = self.model.position - self.target.position
        direction_normalized = glm.normalize(direction)
        self.set_velocity_to_target(-direction_normalized)

        #ignoring Y (XZ plane rotation)
        yaw = glm.atan2(direction_normalized.z, direction_normalized.x)
        self.model.rotation_magnitude.y = -(yaw + pi)

        pitch = 0.5*pi - asin(direction_normalized.y)
        self.model.rotation_magnitude.z = pitch

    @property
    def speed(self):
        return glm.length(self.velocity)
    def set_velocity_to_target(self, direction_normalized):
        """
        Change ship velocity to head towards target
        :return: No return
        """
        self.velocity = direction_normalized * self.speed

    def process_orders(self):
        """
        Read self.orders to change state-machine of ship
        :return:
        """


class Beacon:
    """
    Simple class to give the ships something to fly towards
    """

    def __init__(self, model, position=(0, 0, 0), prices=defaultdict(int), cargo=defaultdict(int), name='Earth'):
        self.position = glm.vec3(position)
        self.model = model
        self.trade_distance = 3.0
        self.prices = prices
        self.cargo = cargo
        self.name = name

    def set_prices(self, prices):
        for good, price in prices.items():
            self.prices[good] = price

    def draw(self, view):
        self.model.draw(view=view)

    def set_position(self, position):
        self.position = glm.vec3(position)
        self.model.position = glm.vec3(position)


class Headquarters:
    def __init__(self, model, wallet=150):
        self.position = model.position
        self.model = model
        self.trade_distance = 3.0
        self.wallet = wallet
        self.data = {}
        self.planet_lookup = {}
        self.name = 'headquarters'
    def draw(self, view):
        self.model.draw(view=view)

    def set_position(self, position):
        self.position = glm.vec3(position)
        self.model.position = glm.vec3(position)

    def buy_ship(
            self,
            price,
            shader,
            texture_dictionary,
            sides=4,
            segments=3,
            transform_x=0.4,
            transform_y=0.4,
            length_of_segment=4.0,
            radius=3.0,
            scale=0.25,
    ):
        self.wallet -= price
        spaceship_model = SpaceshipModel(
            shader=shader,
            # diffuse=texture_dictionary['spaceship_diffuse'],
            # specular=texture_dictionary['spaceship_specular'],
            # emission=texture_dictionary['spaceship_emission'],
            diffuse=texture_dictionary["penguin_diffuse"],
            emission=texture_dictionary["penguin_emission"],
            specular=texture_dictionary["penguin_specular"],
            dimensions=[5.0, 5.0],
            position=self.position,
            rotation_magnitude=[0.0, 0.0, -pi*0.5],
            rotation_axis=glm.vec3((0.0, 0.0, 1.0)),
            number_of_sides=sides,
            number_of_segments=segments,
            transform_x=transform_x,
            transform_z=transform_y,
            length_of_segment=length_of_segment,
            radius=radius,
            scale=scale,
        )
        ship = Spaceship(model=spaceship_model)
        ship.set_velocity(velocity=[1.0, 0.0, 0.0])
        starting_funds = 0
        while starting_funds < 10:
            try:
                starting_funds = float(input(f"Enter Starting Funds (min=10) (wallet={self.wallet})"))
            except(ValueError):
                print("please enter a numerical value")

        ship.wallet = starting_funds
        self.wallet -= starting_funds
        return ship

    def print_info(self):
        """
        print market info of planets
        :return:
        """
        print(f"Wallet: {self.wallet}")
        for planet, prices in self.data.items():
            print(f"{planet}: ", end='')
            for cargo, price in prices.items():
                print(f' {cargo} at ${price} |', end='')
            print()

    #TODO: write this function
    def print_ship_info(self, ships):
        """
        print info on current ships
        :return:
        """
        pass

    def update_planet_lookup(self, new_planet):
        self.planet_lookup[new_planet.name] = new_planet

    def get_planet_lookup(self):
        return self.planet_lookup


