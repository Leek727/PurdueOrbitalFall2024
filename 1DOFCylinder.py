import pygame
import math

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rotating Circle with Firing Thrusters")

# Define colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
FIRING_COLOR = (255, 165, 0)  # Orange to indicate a thruster firing

# Circle properties
circle_pos = (WIDTH // 2, HEIGHT // 2)
circle_radius = 50

# Initial angle of the direction line (in degrees)
angle = 0
rotation_speed = 2  # Speed of rotation in degrees

# Thruster properties (positions around the circle)
thruster_length = 30


def draw_thruster(angle, offset_angle, firing):
    """Draws a single thruster at a specified angle offset from the direction"""
    angle_rad = math.radians(angle + offset_angle)

    # Position of the thruster on the circle's edge
    thruster_start_x = circle_pos[0] + circle_radius * math.cos(angle_rad)
    thruster_start_y = circle_pos[1] - circle_radius * math.sin(angle_rad)

    # End of the thruster vector, indicating thrust direction
    thruster_end_x = thruster_start_x + thruster_length * math.cos(
        angle_rad + math.pi / 2
    )
    thruster_end_y = thruster_start_y - thruster_length * math.sin(
        angle_rad + math.pi / 2
    )

    if firing:
        pygame.draw.line(
            screen,
            (200,200,255),
            (thruster_start_x, thruster_start_y),
            (thruster_end_x, thruster_end_y),
            4,
        )

def draw_thruster2(angle, offset_angle, firing):
    """Draws a single thruster going in the other direction"""
    angle_rad = math.radians(angle + offset_angle)

    # Position of the thruster on the circle's edge
    thruster_start_x = circle_pos[0] + circle_radius * math.cos(angle_rad)
    thruster_start_y = circle_pos[1] - circle_radius * math.sin(angle_rad)

    # End of the thruster vector, indicating thrust direction
    thruster_end_x = thruster_start_x - thruster_length * math.cos(
        angle_rad + math.pi / 2
    )
    thruster_end_y = thruster_start_y + thruster_length * math.sin(
        angle_rad + math.pi / 2
    )

    if firing:
        pygame.draw.line(
            screen,
            (200,200,255),
            (thruster_start_x, thruster_start_y),
            (thruster_end_x, thruster_end_y),
            4,
        )


# Function to draw the circle, direction line, and thrusters
def draw_circle_with_direction(angle, firing_thrusters):
    screen.fill(BLACK)  # Clear the screen

    # Draw the circle
    pygame.draw.circle(screen, WHITE, circle_pos, circle_radius, 2)

    # Calculate the end point of the direction line
    angle_rad = math.radians(angle)
    line_length = circle_radius
    line_end_x = circle_pos[0] + line_length * math.cos(angle_rad)
    line_end_y = circle_pos[1] - line_length * math.sin(angle_rad)

    # Draw the direction line
    pygame.draw.line(screen, RED, circle_pos, (line_end_x, line_end_y), 4)

    # Draw the four thrusters with correct firing state
    draw_thruster(
        angle, 45, firing_thrusters[0]
    )  # Top-right thruster (clockwise rotation)
    draw_thruster(
        angle, 225, firing_thrusters[1]
    )  # Bottom-left thruster (clockwise rotation)
    draw_thruster2(
        angle, 135, firing_thrusters[2]
    )  # Top-left thruster (counterclockwise rotation)
    draw_thruster2(
        angle, 315, firing_thrusters[3]
    )  # Bottom-right thruster (counterclockwise rotation)

    pygame.display.flip()  # Update the display


# --------------------------- dynamics -----------------------------
# cylinder properties
mass = 10  # kg
r = 0.1  # m
I = (mass * (r**2)) / 2  # kg m^2

dt = 1/400

# state vector - [angle, angular velocity]
x = [0, 0]


def update_state(F, state) -> list:
    """Updates state vector one timestep"""
    w = (r * dt * F) / I + state[1]  # angular velocity
    x = [state[0] + w * dt, w]  # Update state vector

    # wrap angle to -pi:pi
    if x[0] > math.pi:
        x[0] -= 2 * math.pi
    elif x[0] < -math.pi:
        x[0] += 2 * math.pi

    return x

def pid_controller(x, setpoint):
    """Will return duty cycle for thrusters"""
    error = setpoint - x[0]
    kP = 1000
    kI = 0 # no friction
    kD = 30

    output = kP * error - kD * x[1]
    
    cap = 100
    if output > cap:
        output = cap
    elif output < -cap:
        output  = -cap
  
    return output

# Main game loop
running = True
clock = pygame.time.Clock()

i = 0
while running:
    i += 1
    firing_thrusters = [False, False, False, False]  # Reset thruster firing states

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # update force
    F = 0

    output = pid_controller(x, 0)
    F = output

    if F < 0:
        firing_thrusters[2] = True
        firing_thrusters[3] = True
    if F > 0:
        firing_thrusters[0] = True 
        firing_thrusters[1] = True
    
    
    # Handle keypresses
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        F = -50

    if keys[pygame.K_RIGHT]:
        F = 50

    # Update state vector
    x = update_state(F, x)

    # update screen
    draw_circle_with_direction(-(x[0] * 360) / (2 * math.pi), firing_thrusters)

    # Limit frame rate
    clock.tick(400)

# Quit Pygame
pygame.quit()
