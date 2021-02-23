import numpy as np


class Gridworld:
    def __init__(self, height, width, goal_position=None, walls=[], traps=[],determinism=1):
        if determinism < 0 or determinism > 1:
            raise ValueError('Determinism should be between 0 and 1')

        self.walls = walls
        self.traps = traps

        self.height = height
        self.width = width
        self.goal_position = goal_position
        trap_num = 2

        if len(self.traps) == 0:
            for _ in range(trap_num):
                ite = 0
                while self.__checkIfGoalorTrapNearby() or len(self.traps) != trap_num:
                    buf = (np.random.randint(1,self.width-1), np.random.randint(1,self.height-1))
                    if not buf in self.traps:
                        self.traps.append(buf)
                    if ite % 10 == 0:
                        self.traps = []
                    ite+=1

        self.determinism = determinism
        
        

        self.max_steps = self.width*self.height - (self.width*2 + (self.height-2)*2)

        # copy goal_position or select randomly
        # cannot be in the same position as a wall or trap
        

        while self.goal_position is None:
            goal_x = np.random.randint(1, self.width - 1)
            goal_y = np.random.randint(1, self.height - 1)
            self.goal_position = (goal_x, goal_y)
            if self.goal_position in self.walls or self.goal_position in self.traps:
                self.goal_position = None
        
        self.step_n = 1
        self.msg = ''

    def __checkIfGoalorTrapNearby(self):
        for y in range(self.height):
            for x in range(self.width):
                if (x,y) in self.traps:
                    if (x+1,y+1) in self.traps or (x,y+1) in self.traps or (x+1,y) in self.traps or (x+1,y-1) in self.traps or (x,y-1) in self.traps or (x-1,y-1) in self.traps or (x-1,y) in self.traps or (x-1,y+1) in self.traps:
                        return True
                    if (x+1,y+1) == self.goal_position or (x,y+1) == self.goal_position or (x+1,y) == self.goal_position or (x+1,y-1) == self.goal_position or (x,y-1) == self.goal_position or (x-1,y-1) == self.goal_position or (x-1,y) == self.goal_position or (x-1,y+1) == self.goal_position or (x,y) == self.goal_position:
                        return True
                    if (x,y) in self.walls:
                        return True
        return False

    def reset(self):
        # generate a random position
        # check if position is empty
        self.agent_position = None
        while self.agent_position is None:
            agent_x = np.random.randint(1, self.width - 1)
            agent_y = np.random.randint(1, self.height - 1)
            self.agent_position = (agent_x, agent_y)
            if (self.agent_position in self.walls or self.agent_position in self.traps or self.agent_position == self.goal_position):
                self.agent_position = None

        return self.agent_position

    def calculate_reward(self, new_state):
        # 10 if agent reached goal
        # -10 if agent is in trap
        # -1 for all other steps
        if new_state == self.goal_position:
            return 10
        if new_state in self.traps:
            return -10
        return - 1

    def is_done(self):
        if self.agent_position == self.goal_position:
            return True
        if self.agent_position in self.traps:
            return True
        if self.step_n == self.max_steps:
            return True
        return False

    def step(self, action):
        # new state, reward, done, info
        # N - 0, E - 1, S - 2, W - 3
        agent_x, agent_y = self.agent_position
        if np.random.uniform(0,1) > self.determinism:
            action = np.random.randint(0,4)
        if action == 0:
            agent_y -= 1
            self.msg = 'N'
        elif action == 1:
            agent_y += 1
            self.msg = 'E'
        elif action == 2:
            agent_x += 1
            self.msg = 'S'
        elif action == 3:
            agent_x -= 1
            self.msg = 'W'
        else:
            raise ValueError('Unknown action', action)

        if agent_x != 0 and agent_x != self.width - 1 and agent_y != 0 and agent_y != self.height - 1 and (agent_x, agent_y) not in self.walls:
            self.agent_position = (agent_x, agent_y)

        reward = self.calculate_reward(self.agent_position)

        done = self.is_done()
        self.step_n += 1
        info = dict()

        return self.agent_position, reward, done, info

    def render(self):
        for y in range(self.height):
            for x in range(self.width):
                if x == 0 or y == 0 or x == self.width-1 or y == self.height-1 or (x,y) in self.walls:
                    print('#', end='')
                elif (x,y) in self.traps:
                    print('o', end='')
                elif (x,y) == self.agent_position:
                    print('A', end='')
                elif (x,y) == self.goal_position:
                    print('X', end='')
                else: 
                    print(' ', end='')
            print()
        print('({})\t({})\n'.format(self.step_n, self.msg))


if __name__ == '__main__':
    world = Gridworld(
        height=5,
        width=5,
        goal_position=(3, 3),
        walls=[(2, 3)],
        determinism=1
    )
    state = world.reset()
    done = False
    while not done:
        world.render()
        action = np.random.randint(0, 4)
        # print(action)
        new_state, reward, done, _ = world.step(action)
        # print("{} - {} -> {}; reward {}; done {}".format(state, action, new_state, reward, done))
        state = new_state
    world.render()
