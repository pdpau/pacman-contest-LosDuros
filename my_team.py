# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Detectar fantasmas y su estado
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        scared_ghosts = [g for g in ghosts if g.scared_timer > 0]
        normal_ghosts = [g for g in ghosts if g.scared_timer <= 0]

        # -- Si hay fantasmas asustados cerca, ir a por ellos -- #
        if len(scared_ghosts) > 0:
            scared_dists = [self.get_maze_distance(my_pos, g.get_position()) for g in scared_ghosts]
            features['ghost_hunt'] = min(scared_dists)
            
            # Penalizar quedarse quieto cerca de fantasmas asustados
            """ if action == Directions.STOP and min_scared_dist <= 3:
                features['stop_near_scared'] = 1 """
            
            # Bonus por moverse hacia el fantasma asustado más cercano
            scared_ghost_pos = scared_ghosts[scared_dists.index(min(scared_dists))].get_position()
            current_dist = self.get_maze_distance(game_state.get_agent_state(self.index).get_position(), scared_ghost_pos)
            if min(scared_dists) < current_dist:
                features['moving_to_scared'] = 1
            
            # Si estamos muy cerca, ignorar otras características y centrarse en la captura
            if min(scared_dists) <= 5:
                features['ghost_hunt'] *= 3  # Triple importancia a la caza
                return features
        # -- -- #

        # -- Si un fantasma está cerca, priorizar la huida -- #
        if len(normal_ghosts) > 0:
            ghosts_dists = [self.get_maze_distance(my_pos, g.get_position()) for g in normal_ghosts]
            if min(ghosts_dists) <= 5:
                features['ghost_distance'] = -10
            if min(ghosts_dists) <= 3:
                if my_state.num_carrying > 0 and len(scared_ghosts) == 0:  
                    features['ghost_distance'] = -50
                    features['distance_to_food'] = 0 # Ignorar comida
        # -- -- #

        # -- Decisión de volver a base -- #
        if self.should_return_home(game_state):
            dist_to_start = self.get_maze_distance(my_pos, self.start)
            features['return_home'] = dist_to_start
            features['distance_to_food'] = 0  # Ignorar comida
            features['capsule_distance'] = 0  # Ignorar cápsulas
        # -- -- #

        # -- Buscar comida -- #
        features['successor_score'] = -len(food_list)
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        # -- -- #

        # -- Buscar cápsulas -- #
        capsules = self.get_capsules(successor)
        capsule_dist = [self.get_maze_distance(my_pos, caps) for caps in capsules]
        if len(capsules) > 0:
            print(f"Capsule distances: {capsule_dist}")
            features['capsule_distance'] = min(capsule_dist)
            # Priorizar cápsulas si hay fantasmas cerca
            if len(normal_ghosts) > 0 and min([self.get_maze_distance(my_pos, g.get_position()) for g in normal_ghosts]) <= 4:
                features['capsule_distance'] *= 3
        else:
            features['will_eat_capsule'] = 1 # Si esta al lado que se coma la capsula
        # -- -- #


        return features

    def get_weights(self, game_state, action):
        # Comprobar si hay fantasmas asustados
        successor = self.get_successor(game_state, action)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        scared_ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None and a.scared_timer > 0]

        # -- Initialize weights variables -- #
        successor_score = 0
        distance_to_food = 0
        ghost_distance = 10
        ghost_hunt = 0
        return_home = 0
        capsule_distance = 0
        moving_to_scared = 0
        # -- -- #

        # Si hay fantasmas asustados, priorizar la caza
        if len(scared_ghosts) > 0:
            distance_to_food = -2
            ghost_distance = 0
            ghost_hunt = -100   # Mayor peso para la persecución
            capsule_distance = -1
            moving_to_scared = 100    # Bonus por acercarse al fantasma

        # Si llevamos comida y no hay fantasmas asustados, priorizar el retorno
        elif self.should_return_home(game_state) and len(scared_ghosts) == 0:
            return_home = -20  # Peso muy alto para volver a casa

        # Comportamiento normal
        else:
            successor_score = 100
            distance_to_food = -3
            ghost_distance = 5
            capsule_distance = -2
    
        return {
            'successor_score': successor_score,
            'distance_to_food': distance_to_food,
            'ghost_distance': ghost_distance,
            'ghost_hunt': ghost_hunt,
            'return_home': return_home,
            #'moving_away': moving_away,
            'capsule_distance': capsule_distance,
            #'stop_near_scared': stop_near_scared,
            'moving_to_scared': moving_to_scared,
            'will_eat_capsule': 1000
        }

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # Remove STOP from actions if there are other options
        if 'Stop' in actions and len(actions) > 1:
            actions.remove('Stop')

        # Debug print
        print(f"\n=== OFFENSIVE Debug Info ===")
        print(f"Legal actions: {actions}")
        values = []
        for a in actions:
            features = self.get_features(game_state, a)
            weights = self.get_weights(game_state, a)
            value = features * weights
            values.append(value)
            # Debug prints
            print(f"\nAction: {a}")
            print(f"Features: {features}")
            print(f"Weights: {weights}")
            print(f"Value: {value}")

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        print(f"Chosen action: {random.choice(best_actions)}\n")
        print(f"-------------------------\n")
        return random.choice(best_actions)
    
    def should_return_home(self, game_state):
        """
        Determina si el agente debe volver a casa basado en múltiples factores
        """
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        carried_food = my_state.num_carrying
        
        # Factores base
        food_left = len(self.get_food(game_state).as_list())
        dist_to_home = self.get_maze_distance(my_pos, self.start)
        time_left = game_state.data.timeleft
        
        # Analizar amenazas
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        ghost_distances = [self.get_maze_distance(my_pos, g.get_position()) for g in ghosts if g.scared_timer <= 0]
        
        # Condiciones de retorno
        must_return = False
        
        # 1. Retorno por tiempo
        time_to_return = dist_to_home + 10  # margen de seguridad
        if time_left <= time_to_return:
            must_return = True
            
        # 2. Retorno por cantidad óptima de comida
        food_threshold = 6  # base
        
        # Ajustar threshold según situación
        if food_left <= 2:  # Casi no queda comida
            food_threshold = 1
        elif dist_to_home > 60 and food_left <= 5:  # Lejos de casa y poca comida
            print("Distance home: ", dist_to_home)
            food_threshold = 8
        elif dist_to_home < 45:
            food_threshold = 4
        if any(d <= 3 for d in ghost_distances):  # Fantasmas cerca
            food_threshold = 2
        elif time_left < 300:  # Poco tiempo
            food_threshold = 2
        
        if carried_food >= food_threshold:
            must_return = True
        
        return must_return


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Mantener posición defensiva
        features['on_defense'] = 1
        if my_state.is_pacman: 
            features['on_defense'] = 0

        # Detectar invasores
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        # Patrol behavior when no invaders
        if len(invaders) == 0:
            patrol_positions = self.get_patrol_positions(game_state)
            
            # Alternate between positions based on time
            patrol_index = (game_state.data.timeleft // 40) % 2
            target_pos = patrol_positions[patrol_index]
            
            patrol_dist = self.get_maze_distance(my_pos, target_pos)
            features['patrol_distance'] = patrol_dist

        # Perseguir invasores agresivamente
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
            
            # Bonus por estar cerca del invasor
            if min(dists) < 2:
                features['close_to_invader'] = 1

        # Evitar quedarse quieto
        if action == Directions.STOP:
            features['stop'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -50,  # Mayor peso para perseguir
            'patrol_distance': -20,  # Importante mantener-nos patrullando
            'close_to_invader': 200,  # Bonus por estar cerca del invasor
            'stop': -100
        }

    def get_boundary_positions(self, game_state): # NOT USED
        """Obtiene las posiciones de la frontera del territorio"""
        boundary = []
        layout = game_state.data.layout
        height = layout.height
        mid_x = (layout.width - 2) // 2
        
        if self.red:
            x = mid_x
        else:
            x = mid_x + 1

        for y in range(1, height - 1):
            if not layout.walls[x][y]:
                boundary.append((x, y))
        
        return boundary

    def get_patrol_positions(self, game_state):
        """
        Creates patrol positions 2 steps back from boundary
        """
        if self.red:
            x = int(game_state.data.layout.width/2 - 2)  # 2 steps back from boundary
        else:
            x = int(game_state.data.layout.width/2 + 2)  # 2 steps back from boundary
            
        patrol_y = []
        height = game_state.data.layout.height
        
        # Find valid y coordinates
        for y in range(1, height - 1):
            if not game_state.has_wall(x, y):
                patrol_y.append(y)
        
        # Return top and bottom third positions
        top_pos = (x, max(patrol_y[:len(patrol_y)//3]))
        bottom_pos = (x, min(patrol_y[-len(patrol_y)//3:]))
        
        return [top_pos, bottom_pos]
