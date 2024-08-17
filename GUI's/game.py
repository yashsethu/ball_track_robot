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
FRAME_RATE = 30

# Set up the game window
with pygame.display.set_mode((WIDTH, HEIGHT)) as screen:
    # Set up the Dino and obstacle
    dino = pygame.image.load("dino.png").get_rect(midleft=(50, DINO_Y))
    obstacle = pygame.image.load("obstacle.png").get_rect(midright=(OBSTACLE_X, DINO_Y))

    # Set up some game variables
    jump = False
    jump_height = 10
    gravity = 1
    obstacle_speed = 5

    # Set up dictionaries for game settings, images, and colors
    settings = {"jump_height": 10, "gravity": 1, "obstacle_speed": 5}

    images = {
        "dino": pygame.image.load("dino.png"),
        "obstacle": pygame.image.load("obstacle.png"),
    }

    colors = {"white": (255, 255, 255)}

    def game_over():
        print("Game Over!")
        pygame.quit()
        sys.exit()

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
            dino.y -= settings["jump_height"]
            settings["jump_height"] -= settings["gravity"]
            if dino.y >= DINO_Y:
                dino.y = DINO_Y
                jump = False
                settings["jump_height"] = 10

        # Move the obstacle
        obstacle.x -= settings["obstacle_speed"]
        if obstacle.x < 0:
            obstacle.x = OBSTACLE_X

        # Check for collision
        if dino.colliderect(obstacle):
            game_over()

        # Draw everything
        screen.fill(colors["white"])
        screen.blit(images["dino"], dino)
        screen.blit(images["obstacle"], obstacle)
        pygame.display.flip()

        # Cap the frame rate
        pygame.time.Clock().tick(FRAME_RATE)
