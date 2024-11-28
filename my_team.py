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

        # Si hay fantasmas asustados cerca, ir a por ellos
        if len(scared_ghosts) > 0:
            scared_dists = [self.get_maze_distance(my_pos, g.get_position()) for g in scared_ghosts]
            min_scared_dist = min(scared_dists)
            features['ghost_hunt'] = min_scared_dist
            
            # Penalizar quedarse quieto cerca de fantasmas asustados
            if action == Directions.STOP and min_scared_dist <= 3:
                features['stop_near_scared'] = 1
            
            # Bonus por moverse hacia el fantasma asustado más cercano
            scared_ghost_pos = scared_ghosts[scared_dists.index(min_scared_dist)].get_position()
            current_dist = self.get_maze_distance(game_state.get_agent_state(self.index).get_position(), scared_ghost_pos)
            if min_scared_dist < current_dist:
                features['moving_to_scared'] = 1
            
            # Si estamos muy cerca, ignorar otras características y centrarse en la captura
            if min_scared_dist <= 2:
                features['ghost_hunt'] *= 2  # Doble importancia a la caza
                return features

        # Comportamiento normal cuando no hay fantasmas asustados
        if len(normal_ghosts) > 0:
            dists = [self.get_maze_distance(my_pos, g.get_position()) for g in normal_ghosts]
            min_ghost_dist = min(dists)
            
            if min_ghost_dist <= 3:
                features['ghost_distance'] = -10
                if my_state.num_carrying > 0 and len(scared_ghosts) == 0:  
                    features['ghost_distance'] = -20

        # Decisión de volver a base
        carrying_threshold = 3
        if my_state.num_carrying >= carrying_threshold:
            # Solo volver si no hay fantasmas asustados cerca
            if len(scared_ghosts) == 0 or min([self.get_maze_distance(my_pos, g.get_position()) for g in scared_ghosts]) > 5:
                dist_to_start = self.get_maze_distance(my_pos, self.start)
                features['return_home'] = dist_to_start
                if dist_to_start > self.get_maze_distance(game_state.get_agent_state(self.index).get_position(), self.start):
                    features['moving_away'] = 1
                return features

        # Buscar comida
        features['successor_score'] = -len(food_list)
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # Buscar cápsulas
        capsules = self.get_capsules(successor)
        if len(capsules) > 0:
            min_capsule_dist = min([self.get_maze_distance(my_pos, caps) for caps in capsules])
            features['capsule_distance'] = min_capsule_dist
            # Priorizar cápsulas si hay fantasmas cerca
            if len(normal_ghosts) > 0 and min([self.get_maze_distance(my_pos, g.get_position()) for g in normal_ghosts]) <= 4:
                features['capsule_distance'] *= 2

        return features

    def get_weights(self, game_state, action):
        # Comprobar si hay fantasmas asustados
        successor = self.get_successor(game_state, action)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        scared_ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None and a.scared_timer > 0]

        # Si hay fantasmas asustados, priorizar la caza
        if len(scared_ghosts) > 0:
            return {
                'successor_score': 100,
                'distance_to_food': -2,
                'ghost_distance': 0,
                'ghost_hunt': -50,    # Mayor peso para la persecución
                'capsule_distance': -1,
                'stop_near_scared': -200,  # Fuerte penalización por quedarse quieto
                'moving_to_scared': 100    # Bonus por acercarse al fantasma
            }
        
        # Si llevamos comida y no hay fantasmas asustados, priorizar el retorno
        if game_state.get_agent_state(self.index).num_carrying >= 3 and len(scared_ghosts) == 0:
            return {
                'return_home': -100,  # Peso muy alto para volver a casa
                'moving_away': -1000,  # Penalización muy alta por alejarse
            }

        # Comportamiento normal
        return {
            'successor_score': 100,
            'distance_to_food': -2,
            'ghost_distance': 10,
            'capsule_distance': -10,
        }


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

        # Posicionamiento en la frontera cuando no hay invasores
        if len(invaders) == 0:
            boundary = self.get_boundary_positions(game_state)
            min_boundary_dist = min([self.get_maze_distance(my_pos, pos) for pos in boundary])
            features['boundary_distance'] = min_boundary_dist
        
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
            'boundary_distance': -20,  # Importante mantener posición en frontera
            'close_to_invader': 200,  # Bonus por estar cerca del invasor
            'stop': -100
        }

    def get_boundary_positions(self, game_state):
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
