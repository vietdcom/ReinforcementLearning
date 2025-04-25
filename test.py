import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font(None, 25)

# Direction enum
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Point structure
Point = namedtuple('Point', 'x, y')

# Game config
BLOCK_SIZE = 20
SPEED = 10

# Snake game class
class SnakeGameAI:
    def __init__(self, w=400, h=400):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w//2, self.h//2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake:
                break

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)  # Update the head
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        if self._is_collision():
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        return (
            pt.x >= self.w or pt.x < 0 or
            pt.y >= self.h or pt.y < 0 or
            pt in self.snake[1:]
        )

    def _update_ui(self):
        self.display.fill((0, 0, 0))
        for pt in self.snake:
            pygame.draw.rect(self.display, (0, 255, 0), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, (255, 0, 0), pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render("Score: " + str(self.score), True, (255, 255, 255))
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # straight
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]  # right
        else:
            new_dir = clock_wise[(idx - 1) % 4]  # left

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def get_state(self):
        head = self.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        danger_straight = (dir_r and self._is_collision(point_r)) or \
                          (dir_l and self._is_collision(point_l)) or \
                          (dir_u and self._is_collision(point_u)) or \
                          (dir_d and self._is_collision(point_d))

        danger_right = (dir_u and self._is_collision(point_r)) or \
                       (dir_d and self._is_collision(point_l)) or \
                       (dir_l and self._is_collision(point_u)) or \
                       (dir_r and self._is_collision(point_d))

        danger_left = (dir_u and self._is_collision(point_l)) or \
                      (dir_d and self._is_collision(point_r)) or \
                      (dir_l and self._is_collision(point_d)) or \
                      (dir_r and self._is_collision(point_u))

        food_left = self.food.x < head.x
        food_right = self.food.x > head.x
        food_up = self.food.y < head.y
        food_down = self.food.y > head.y

        state = [
            danger_straight,
            danger_right,
            danger_left,
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            food_left,
            food_right,
            food_up,
            food_down
        ]

        return np.array(state, dtype=int)

# --- MANUAL CONTROL TEST LOOP ---
# --- MANUAL CONTROL TEST LOOP (WASD) ---
if __name__ == '__main__':
    game = SnakeGameAI()

    direction_map = {
        pygame.K_w: Direction.UP,
        pygame.K_s: Direction.DOWN,
        pygame.K_a: Direction.LEFT,
        pygame.K_d: Direction.RIGHT
    }

    while True:
        action = [1, 0, 0]  # default to straight
        keys = pygame.key.get_pressed()

        for key, new_dir in direction_map.items():
            if keys[key]:
                # Only allow turning if it's not directly opposite
                if (game.direction == Direction.LEFT and new_dir != Direction.RIGHT) or \
                   (game.direction == Direction.RIGHT and new_dir != Direction.LEFT) or \
                   (game.direction == Direction.UP and new_dir != Direction.DOWN) or \
                   (game.direction == Direction.DOWN and new_dir != Direction.UP):
                    game.direction = new_dir
                break

        # Compute action based on new direction
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(game.direction)
        current_idx = clock_wise.index(game.direction)

        if game.direction == clock_wise[current_idx]:
            action = [1, 0, 0]  # straight

        reward, game_over, score = game.play_step(action)

        if game_over:
            print('Final Score', score)
            game.reset()