import random
import pygame
from itertools import product

import matplotlib.pyplot as plt


# NOTE:
# - per iterare su tutti gli stati senza cicli for annidati, si
#   consiglia di usare la funzione product (dal modulo itertools, già
#   importata) nel modo seguente:
#   "for s in product(range(w), range(h))"
#   oppure
#   "for x, y in product(range(w), range(h))"
#   dove w e h sono le dimensioni della griglia.


def value_iteration(
    width,
    height,
    num_actions,
    transition_probabilities,
    rewards,
    end_state,
    max_steps,
    gamma,
    iterations
):
    """Implementare Value Iteration.

    Parametri:
    - width e height: larghezza e altezza della griglia
    - num_actions: numero di azioni disponibili (sempre 4 in questa
      esercitazione)
    - transition_probabilities: dict tale che
      transition_probabilities[s, a][s1] è la probabilità di arrivare
      nello stato s1, dato lo stato di partenza s e l'azione a.
    - rewards: dict tale che rewards[s1] è il reward ottenuto quando una
      transizione finisce nello stato s1
    - end_state: stato di goal ((4,3) in questa esercitazione)
    - max_steps: massimo numero di step che l'environment ammette per
      ogni episodio
    - gamma: discount factor
    - iterations: numero di iterazioni di Value Iteration da eseguire

    Valori di ritorno:
    - values: dict tale che values[s] è la stima di V*(s)
    - best_actions: dict tale che best_actions[s] = pi*(s)
    """
    # Initialize V*(s) to 0 for all s and pi*(s) to any action (0 in this case).
    values = {s: 0.0 for s in product(range(width), range(height))}
    best_actions = {s: 0 for s in product(range(width), range(height))}
    
    # TODO completare

    return values, best_actions

def q_learning(
    width,
    height, 
    num_actions,
    alpha,
    gamma,
    initial_epsilon,
    epsilon_decay,
    episodes,
    eval_ql_policy,
):
    """Implementare Q-Learning.

    Parametri:
    - width e height: larghezza e altezza della griglia
    - num_actions: numero di azioni disponibili (sempre 4 in questa
      esercitazione)
    - alpha: learning rate
    - gamma: discount factor
    - initial_epsilon: epsilon iniziale per la policy epsilon-greedy
    - epsilon_decay: valore del quale va scalato epsilon dopo ogni
      episodio di training, a partire dal primo episodio in cui si
      ottiene un reward non nullo
    - episodes: numero di episodi di training
    - eval_ql_policy: funzione da usare come eval_ql_policy(q_table),
      che valuta la q_table (vedere valori di ritorno) eseguendo un
      episodio sull'environment, ritornando il reward cumulativo
      ottenuto

    Valori di ritorno:
    - q_table: dict tale che q_table[s, a] ritorna la stima Q*(s, a)
    - evals: lista di lunghezza uguale ad episodes, contenente il reward
      cumulativo medio ottenuto usando almeno 10 ripetizioni di
      eval_ql_policy(q_table) dopo ogni episodio di training     
    """
    evals = []
    q_table = {}
    for s, a in product(product(range(width), range(height)), range(num_actions)):
        q_table[s, a] = 0
    eps = initial_epsilon
    received_first_reward = False

    # TODO completare

    return q_table, evals




# Implementation (grid world, rendering, main)

class GridWorld:
    def __init__(
        self,
        width,
        height,
        start,
        goal,
        solid_cells,
        max_steps,
        p_diag
    ) -> None:
        self.width = width
        self.height = height
        self.goal = goal
        self.start = start
        self.max_steps = max_steps
        self.p_diag = p_diag
        if (0, 0) in solid_cells or goal in solid_cells:
            raise ValueError(f'Start or goal cells cannot be solid.')
        self.solid_mask = {}
        for x in range(width):
            for y in range(height):
                self.solid_mask[(x, y)] = (x, y) in solid_cells
        self.action_deltas = [
            (-1, 0), # left
            (0, -1), # up
            (1, 0), # right
            (0, 1), # down
        ]
        self._init_transition_probabilities()
        self._init_rewards()

    def _init_transition_probabilities(self):
        self.transition_probabilities = {}
        for s, a in product(product(range(self.width), range(self.height)), range(4)):
            self.transition_probabilities[s, a] = {
                s1: 0. for s1 in product(range(self.width), range(self.height))
            }
            s1 = self.deterministic_transition(s, a)
            s2 = self.deterministic_transition(s1, (a+1)%4)
            _action_arrows = ['←','↑','→','↓']
            #print(f'Probs for {s}, {a} ({_action_arrows[a]}):')
            self.transition_probabilities[s, a][s1] += 1. - self.p_diag
            self.transition_probabilities[s, a][s2] += self.p_diag
            #print(f'{self.transition_probabilities[s, a]}')

    def _init_rewards(self):
        self.rewards = {s: 1. if s == self.goal else 0. for s in product(range(self.width), range(self.height)) }

    def reset(self):
        self.steps = 0
        self.state = self.start
        return self.state

    def deterministic_transition(self, state, action):
        delta = self.action_deltas[action]
        vplus = lambda v, w: (v[0] + w[0], v[1] + w[1])
        s1 = vplus(state, delta)
        if s1[0] < 0 or s1[0] >= self.width:
            return state
        if s1[1] < 0 or s1[1] >= self.height:
            return state
        if self.solid_mask[s1]:
            return state
        return s1

    def step(self, action):
        self.state = self.deterministic_transition(self.state, action)
        if random.random() < self.p_diag:
            self.state = self.deterministic_transition(self.state, (action+1)%4)
        success = self.state == self.goal
        reward = 1. if success else 0.
        self.steps += 1
        truncated = self.steps == self.max_steps
        return self.state, reward, success or truncated

    def render(self, screen: pygame.Surface) -> None:
        w, h = screen.get_width(), screen.get_height()
        grid_width = self.width
        grid_height = self.height
        padding_cells = 1
        cell_width = int(w / (2*padding_cells + grid_width))
        cell_height = int(h / (2*padding_cells + grid_height))
        x_pad, y_pad = cell_width*padding_cells, cell_height*padding_cells
        agent_size = goal_size = (2/3)*min(cell_height, cell_width)
        # clear the screen
        screen.fill("white")
        # draw the gray cells
        for x_gray in range(self.width):
            for y_gray in range(self.height):
                if not self.solid_mask[(x_gray, y_gray)]:
                    continue
                pygame.draw.rect(
                    screen,
                    color="gray",
                    rect=pygame.Rect(
                        x_pad + x_gray*cell_width,
                        y_pad + y_gray*cell_height,
                        cell_width,
                        cell_height
                    )
                )
        # draw the start cell
        pygame.draw.rect(
            screen,
            color="yellow",
            rect=pygame.Rect(
                    x_pad,
                    y_pad,
                    cell_width,
                    cell_height
            )
        )
        # draw the goal cell
        pygame.draw.rect(
            screen,
            color="green",
            rect=pygame.Rect(
                    x_pad + self.goal[0]*cell_width,
                    y_pad + self.goal[1]*cell_height,
                    cell_width,
                    cell_height
            )
        )
        # draw the grid borders
        pygame.draw.rect(
            screen,
            color="black",
            rect=pygame.Rect(x_pad, y_pad, cell_width*grid_width, cell_height*grid_height),
            width=1,
        )
        # draw cell borders
        for i in range(grid_width-1):
            pygame.draw.line(
                screen,
                color="black",
                start_pos=[x_pad + (i+1)*cell_width, y_pad],
                end_pos=[x_pad + (i+1)*cell_width, y_pad + grid_height*cell_height]
            )
        for i in range(grid_height-1):
            pygame.draw.line(
                screen,
                color="black",
                start_pos=[x_pad, y_pad + (i+1)*cell_height],
                end_pos=[x_pad + grid_width*cell_width, y_pad + (i+1)*cell_height]
            )
        # draw the agent
        x_agent_screen = cell_width/2 + (padding_cells + self.state[0])*cell_width
        y_agent_screen = cell_height/2 + (padding_cells + self.state[1])*cell_height
        pygame.draw.circle(
            screen,
            color="blue",
            center=[x_agent_screen, y_agent_screen],
            radius=agent_size/2,
        )


def run(env, policy):

    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()
    running = True
    frames_per_step = 30 # determines speed of rendering
    frames = 1

    obs = env.reset()
    done = False
    reward = 0.
    ret = 0.
    env.render(screen=screen)

    _action_arrows = [ '←', '↑', '→', '↓' ]

    while running and not done:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # advance world by one time step and render it to screen
        if frames % frames_per_step == 0:
            action = policy(obs)
            print(f'Executing {action} ({_action_arrows[action]}) from {obs}')
            obs, reward, done = env.step(action)
            ret += reward
            env.render(screen=screen)
        frames += 1

        # flip() the display to put your work on screen
        pygame.display.flip()

        clock.tick(60) # limits FPS to 60

    pygame.quit()

    print(f'Return (sum of rewards): {ret}.')


def print_grid(width, height, vals, title=None, sep=' '):
    if title is not None:
        print(title)
    for c in range(height):
        for r in range(width):
            v = vals[r, c]
            print(f'{sep}{v}', end='')
        print()
    print()

def cell_str(env, x, y):
    if env.solid_mask[(x,y)]:
        return '⊗'
    elif (x,y) == env.start:
        return '⊚'
    elif (x,y) == env.goal:
        return '⊛'
    else:
        return '⊙'


if __name__ == '__main__':

    action_arrows = ['←', '↑', '→', '↓' ]

    random.seed(44)
    env = GridWorld(
        width=5,
        height=4,
        start=(0,0),
        goal=(4,3),
        solid_cells=[(1,0),(2,1),(2,3)],
        max_steps=100,
        p_diag=0.3,
    )

    print(f'probs for (0,0), right: {env.transition_probabilities[(0,0), 2]}', )

    print_grid(
        env.width,
        env.height,
        { s: cell_str(env, s[0], s[1]) for s in product(range(env.width), range(env.height)) },
        title='Env map:'
    )

    print_grid(
        env.width,
        env.height,
        { s: env.rewards[s] for s in product(range(env.width), range(env.height)) },
        title='Rewards:'
    )

    print(f'Exercise 1: Value iteration')
    values, best_actions = value_iteration(
        width=env.width,
        height=env.height,
        num_actions=4,
        transition_probabilities=env.transition_probabilities,
        rewards=env.rewards,
        end_state=env.goal,
        max_steps=env.max_steps,
        gamma=0.99,
        iterations=100,
    )

    print_grid(
        env.width,
        env.height,
        { s: f'{values[s]:.2f}' for s in product(range(env.width), range(env.height)) },
        title=f'Values obtained with 100 iterations of value iteration:'
    )

    print_grid(
        env.width,
        env.height,
        { s: action_arrows[best_actions[s]] for s in product(range(env.width), range(env.height)) },
        title=f'Best actions obtained with 100 iterations of value iteration:'
    )

    print(f'Running policy obtained with value iteration')
    run(env, lambda s: best_actions[s])

    def argmax(f, xs):
        x_max = None
        f_max = None
        for x in xs:
            y = f(x)
            if f_max is None or y > f_max:
                x_max = x
                f_max = y
        return x_max

    def eval_ql_policy(q):
        s = env.reset()
        done = False
        ret = 0.
        while not done:
            action = argmax(lambda a: q[s, a], range(4))
            s, rew, done = env.step(action)
            ret += rew
        return ret

    print(f'Exercise 2: Q-Learning')
    ql_fixed_params = dict(
        width=env.width,
        height=env.height,
        num_actions=4,
        gamma=0.99,
        initial_epsilon=1.0,
        epsilon_decay=0.99,
        episodes=20,
        eval_ql_policy=eval_ql_policy
    )

    q_table_01, ql_evals_01 = q_learning(**ql_fixed_params, alpha=0.1)
    q_table_05, ql_evals_05 = q_learning(**ql_fixed_params, alpha=0.5)
    q_table_1, ql_evals_1 = q_learning(**ql_fixed_params, alpha=1.0)

    print_grid(
        env.width,
        env.height,
        { s: f'{max(q_table_01[s, a] for a in range(4)):.2f}' for s in product(range(env.width), range(env.height)) },
        title=f'Values obtained with {ql_fixed_params["episodes"]} episodes of Q-Learning and alpha = 0.1:'
    )

    print_grid(
        env.width,
        env.height,
        { s: action_arrows[ argmax(lambda a: q_table_01[s, a], range(4)) ] for s in product(range(env.width), range(env.height)) },
        title=f'Best actions obtained with {ql_fixed_params["episodes"]} episodes of Q-Learning and alpha = 0.1:'
    )

    print(f'Running policy from experiment with alpha = 0.1')
    run(env, lambda s: argmax(lambda a: q_table_01[s, a], range(4)))

    plt.xticks(list(range(1, ql_fixed_params["episodes"]+1)))
    plt.plot(list(range(1, ql_fixed_params["episodes"]+1)), ql_evals_01, label="alpha = 0.1")
    plt.plot(list(range(1, ql_fixed_params["episodes"]+1)), ql_evals_05, label="alpha = 0.5")
    plt.plot(list(range(1, ql_fixed_params["episodes"]+1)), ql_evals_1, label="alpha = 1")
    plt.xlabel("Episodes trained")
    plt.ylabel("Average eval return")
    plt.legend()
    plt.show()

