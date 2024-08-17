import pygame
import sys
import RPi.GPIO as GPIO

# Set up the GPIO pins
GPIO.setmode(GPIO.BCM)
GPIO.setup(16, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(22, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(24, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Initialize Pygame
pygame.init()

# Set up some constants
WIDTH, HEIGHT = 800, 600
DINO_HEIGHT = 50
OBSTACLE_WIDTH = 20
OBSTACLE_HEIGHT = 50
DINO_Y = HEIGHT - DINO_HEIGHT
OBSTACLE_X = WIDTH

# Set up the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Set up the Dino and obstacle
dino = pygame.Rect(50, DINO_Y, 50, DINO_HEIGHT)
obstacle = pygame.Rect(OBSTACLE_X, DINO_Y, OBSTACLE_WIDTH, OBSTACLE_HEIGHT)

# Set up some game variables
jump = False
jump_height = 10
gravity = 1
obstacle_speed = 5

# Game loop
while True:
    # Event loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Check if the up button is pressed
    if GPIO.input(16) == GPIO.LOW and dino.y == DINO_Y:
        jump = True

    # Make the Dino jump
    if jump:
        dino.y -= jump_height
        jump_height -= gravity
        if dino.y >= DINO_Y:
            dino.y = DINO_Y
            jump = False
            jump_height = 10

    # Move the obstacle
    obstacle.x -= obstacle_speed
    if obstacle.x < 0:
        obstacle.x = OBSTACLE_X

    # Check for collision
    if dino.colliderect(obstacle):
        print("Game Over!")
        pygame.quit()
        sys.exit()

    # Draw everything
    screen.fill((255, 255, 255))
    pygame.draw.rect(screen, (0, 0, 0), dino)
    pygame.draw.rect(screen, (0, 0, 0), obstacle)
    pygame.display.flip()

    # Cap the frame rate
    pygame.time.Clock().tick(30)
