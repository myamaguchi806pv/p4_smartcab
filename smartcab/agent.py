import random
import math
import operator
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

# Q value table
q_table = {}


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.state = None

        # previous state and action
        self.previous_state = None
        self.previous_action = None
        self.previous_reward = 0.0
        self.alpha=1.0 # learning rate
        self.gamma=0.1 # discount factor


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.calcState(inputs, self.next_waypoint)
        # print 'Update State'
        # print self.state

        # update Q value
        self.updateQtable(self.previous_state, self.previous_action, self.state, self.previous_reward)

        # TODO: Select action according to your policy
        #random action policy
        # action = self.RandomChoice()
        action = self.greedyChoice(self.state)
        # action = self.epsilonGreedyChoice(self.state, 0.5)
        # action = self.softmaxChoice(self.state, tau= 1.0)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

        # record state, action and reward to update Q value in next step
        self.previous_state = self.state
        self.previous_action = action
        self.previous_reward = reward

        # print "LearningAgent.update(): deadline = {}, state = {}, action = {}, reward = {}".format(deadline, self.state, action, reward)  # [debug]

    def updateQtable(self, state, action, next_state, reward):
        old_q_value = self.getQValue(state, action)

        new_q_value = old_q_value + self.alpha * ( reward + self.gamma *self.getMaxQvalue(next_state) - old_q_value)

        self.setQValue(state, action, new_q_value)

        # print "Updated Q Table:"
        # print q_table
        return

    def getQValue(self, state, action):
        try:
            str_state = str(state)
            # print "Get Q Value:" + str()
            return q_table[str_state][action]
        except KeyError:
            # print "Get Q Value: KeyError " + str(0.0)
            return 0.0

    def setQValue(self, state, action, q_value):
        str_state = str(state)
        # q_table.setdefault(str_state,{})
        q_table.setdefault(str_state,{None: 0.0,'forward': 0.0, 'right': 0.0, 'left':0.0})
        q_table[str_state][action] = q_value
        # print "set Q({}, {}) = {}".format(str_state, action, q_value)

    def getQActions(self, state):
        str_state = str(state)
        return q_table.setdefault(str_state,{None: 0.0,'forward': 0.0, 'right': 0.0, 'left':0.0})

    def getMaxQvalue(self, state):
        str_state = str(state)
        # print "Max Q Value in " + str([self.getQValue(str_state, act) for act in self.env.valid_actions] ) + " is "+ str(max( [self.getQValue(str_state, act) for act in self.env.valid_actions] ))
        return max( [self.getQValue(str_state, act) for act in self.env.valid_actions] )

    def greedyChoice(self, state):
        str_state=str(state)
        try:
            # print 'Actions: ' + str(q_table[str_state])
            max_q_value = max(q_table[str_state].values())
            max_action_indexes = [i for i, x in enumerate(q_table[str_state].values()) if x == max_q_value]
            # print " Max action indexes: " + str(max_action_indexes)
            ret = q_table[str_state].keys()[random.choice(max_action_indexes)]
            # print "return Action: " + str(ret)
            # ret = max(q_table[str_state].iteritems(), key=operator.itemgetter(1))[0]
        except KeyError:
            ret = random.choice(self.env.valid_actions)
        # print "argmax action:" + str(ret)
        return ret

    def epsilonGreedyChoice(self, state, epsilon=0.0):
        if random.random() <= epsilon:
            return self.RandomChoice()
        else:
            return self.greedyChoice(state)

    def RandomChoice(self):
        return random.choice(self.env.valid_actions)

    def softmaxChoice(self, state, tau):
        actions = self.getQActions(state)
        # print "Actions: " + str(actions)
        temp = actions.copy()
        for k, v in temp.items():
            temp[k] = math.exp(v)/tau
        # print "Temp: " + str(temp)
        weight = temp.copy()
        for k, v in temp.items():
            weight[k] = v/sum(temp.values())
        # print "Weight: " + str(weight)
        action = self.weightChoice(weight)
        return action

    def weightChoice(self, weight):
        random_value = random.random()
        for i in range(len(weight)):
            if random_value < sum(weight.values()[:i+1]):
                return weight.keys()[i]

    def calcState(self, inputs, next_waypoint):
        # print "This is inputs: {}".format(str(inputs)) #{'light': 'red', 'oncoming': 'left', 'right': None, 'left': None}

        # make state from available directions and next waypoint
        state = {'light': inputs["light"], 'oncoming':inputs["oncoming"], 'right':inputs["right"],'left':inputs["left"], 'next_waypoint':next_waypoint}
        return state



def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    # sim = Simulator(e, update_delay=0.1, display=True)  # create simulator (uses pygame when display=True, if available)
    # sim = Simulator(e, update_delay=0.001, display=False)
    sim = Simulator(e, update_delay=0.001, display=True, live_plot=True)

    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
