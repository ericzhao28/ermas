import numpy as np
import random

state_dim = 7
max_timesteps = 1000  # max timesteps in one episode

season_period = 2 * 365 / (2 * 3.14)  # approx two year s
consumption_max = 200  # 20 me's consuming 10 each a day
seasonal_spending = lambda t: consumption_max * (np.cos(t / season_period) + 1
                                                 ) / 2
consumer_inventory_max = 2000  # 20 fridges can hold 2000 macarons
supplier_inventory_cost = 0.05  # 5 cents a day; can hold for half a year at $30
supplier_prod_inc = 8  # 80 macarons a day maximum
supplier_prod_num = 10
consumer_buy_inc = 50  # Buy up to 500 a day
consumer_buy_num = 10
market_price_inc = 1  # Macarons cost up to $10 each (maxed out by happiness)
market_price_num = 11
shipping_cost_inc = 10  # $100/day shipping cost; if they buy $2000 a day, negligible
shipping_cost_num = 10

action_dim_p = shipping_cost_num
action_dim_a1 = market_price_num * supplier_prod_num
action_dim_a2 = consumer_buy_num


def parse_supplier_action(action):
    market_price_a = int(action / supplier_prod_num)
    supplier_prod_a = action % supplier_prod_num
    return market_price_a, supplier_prod_a


class TradingEnv():
    def __init__(self, consumer_happiness=10):
        self.consumer_happiness = consumer_happiness  # each macaron is ~ $10/happiness
        self.consumer_happiness_random = False

    def reset(self):
        # Wealth
        self.consumer_wealth = 0
        self.shipping_wealth = 0
        self.supplier_wealth = 0

        # Inventory
        self.market_inventory = 0
        self.consumer_inventory = 0

        # Prices
        self.market_price = 0
        self.next_market_price = None
        self.shipping_price = 0

        # Time
        self.counter = 0

        return self.get_state()

    def get_state(self):
        return np.array([
            self.market_inventory,
            self.consumer_inventory,
            self.market_price,
            self.next_market_price or -1,
            self.shipping_price,
            self.counter / max_timesteps,
            seasonal_spending(self.counter + 1),
        ],
                        dtype=np.float32)

    def get_stats(self):
        return {
            "market inventory": self.market_inventory,
            "consumer inventory": self.consumer_inventory,
            "market price": self.market_price,
            "next market price": self.next_market_price or -1,
            "shipping price": self.shipping_price,
            "seasonal spending": seasonal_spending(self.counter + 1),
            "consumer wealth": self.consumer_wealth,
            "supplier wealth": self.supplier_wealth,
            "shipping wealth": self.shipping_wealth,
            "consumer bought": self.consumer_bought,
            "consumer spent": self.consumer_spent,
        }

    def step(self, actions):
        ship_action, s_action, c_action = actions

        ### Translate actions
        c_purchases = c_action * consumer_buy_inc
        market_price_a, supplier_prod_a = parse_supplier_action(s_action)
        new_market_price = market_price_a * market_price_inc
        supplier_prod = supplier_prod_a * supplier_prod_inc
        new_shipping_price = ship_action * shipping_cost_inc

        ### Remember old values
        old_shipping_wealth = self.shipping_wealth
        old_supplier_wealth = self.supplier_wealth
        old_consumer_wealth = self.consumer_wealth

        ### Handle consumer purchases
        self.consumer_bought = min(
            consumer_inventory_max - self.consumer_inventory,
            min(self.market_inventory, c_purchases))
        # Update inventories
        self.market_inventory -= self.consumer_bought
        self.consumer_inventory += self.consumer_bought
        # Handle product costs
        self.supplier_wealth += self.consumer_bought * self.market_price
        self.consumer_wealth -= self.consumer_bought * self.market_price
        # Handle shipping costs
        self.shipping_wealth += np.sqrt(
            self.consumer_bought) * self.shipping_price
        self.supplier_wealth -= np.sqrt(
            self.consumer_bought) * self.shipping_price / 2
        self.consumer_wealth -= np.sqrt(
            self.consumer_bought) * self.shipping_price / 2

        ### Market updates
        # Update market prices
        if self.next_market_price is not None:
            self.market_price = self.next_market_price
        self.next_market_price = new_market_price
        # Update supplier inventory
        self.market_inventory += supplier_prod
        self.supplier_wealth -= self.market_inventory * supplier_inventory_cost
        # Update shipping
        self.shipping_price = new_shipping_price

        ### Update consumers
        self.consumer_spent = min(self.consumer_inventory,
                                  seasonal_spending(self.counter))
        self.consumer_inventory -= self.consumer_spent
        if self.consumer_happiness_random:
            self.consumer_wealth += self.consumer_spent * (10 + self.consumer_happiness_random * (random.random() - 0.5))
        else:
            self.consumer_wealth += self.consumer_spent * self.consumer_happiness

        ### Handle rewards
        ship_reward = self.shipping_wealth - old_shipping_wealth
        s_reward = self.supplier_wealth - old_supplier_wealth
        c_reward = self.consumer_wealth - old_consumer_wealth
        rewards = (ship_reward, s_reward, c_reward)

        ### Handle timer
        done = self.counter > max_timesteps
        self.counter += 1

        return self.get_state(), rewards, done

    def seed(self, seed_num):
        np.random.seed(seed_num)
        random.seed(seed_num)
