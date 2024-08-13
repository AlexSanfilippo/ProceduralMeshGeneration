"""
4th February, 2024
last update: 4th August, 2024
This files contains a code for creating and displaying our proc-gen spaceships

-[V]upload to github(initial)
-[V]Emission Map in our default shader
-[V]clear out old WFC code
-[V]basic spaceship generator (in primative_meshes)
    MAJOR FEATURES
    -[V]basic segmented prism (main section)
        -[]Issue: Extrude is flipping texture around
            -I think we need to re-calcuate texture coordinates of new-face
                -how to go from outer-coordinates to texture coordinates?
    -[V]make main section symmetrical
    -[V]front/cockpit/nose
    -[V]rear thrusters
        -[V]glowing (emission map)
    -all one-mesh model
    -all in one texture  dictionary

-[]some sort of propulsion trail on our ship
-[V]a starry skybox around the ship
-[]share and post on reddit, discord
-[]a video documenting our process

"""
import math
import threading
from collections import defaultdict

import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr

import ProceduralMesh
from TextureLoader import load_texture
from Camera import Camera, FollowCamera
import math as m
import random as r
from random import random
import numpy as np
import glm
from PIL import Image
import PointLightCube as plc
import ProceduralMesh as primatives
import FPSCounter
import SpaceShip
from Menu import Menu
from SpaceShip import Headquarters

#AUDIO
from pydub import AudioSegment
from pydub.playback import play


images = []

follow_cam = FollowCamera(camera_pos=[6.0, 17.0, 12.0])
cam = Camera(camera_pos=[0.0, 20.0, 20.0])
WIDTH, HEIGHT = 1280, 720
lastX, lastY = WIDTH / 2, HEIGHT / 2
first_mouse = True
left, right, forward, backward, make_new_surface = False, False, False, False, False
player_left, player_right, player_forward, player_backward = False, False, False, False
camera_mode = 0 #1 for follow, 0 for fly
cycle_camera_mode = False
yaw_counterclockwise, yaw_clockwise = False, False
write_to_gif = False
make_new_ship = False
pause = False


"""MENU TESTING"""
my_menu = Menu()
# the keyboard input callback
def key_input_clb(window, key, scancode, action, mode):
    global left, right, forward, backward, make_new_surface, player_left, player_right, player_forward, \
        player_backward, cycle_camera_mode, yaw_counterclockwise, yaw_clockwise, write_to_gif, make_new_ship,\
        pause

    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)
    if key == glfw.KEY_LEFT and action == glfw.PRESS:
        player_left = True
    if key == glfw.KEY_RIGHT and action == glfw.PRESS:
        player_right = True
    if key == glfw.KEY_UP and action == glfw.PRESS:
        player_forward = True
    if key == glfw.KEY_DOWN and action == glfw.PRESS:
        player_backward = True
    if key == glfw.KEY_E and action == glfw.PRESS:
        make_new_surface = True
    if key == glfw.KEY_W and action == glfw.PRESS:
        forward = True
    elif key == glfw.KEY_W and action == glfw.RELEASE:
        forward = False
    if key == glfw.KEY_S and action == glfw.PRESS:
        backward = True
    elif key == glfw.KEY_S and action == glfw.RELEASE:
        backward = False
    if key == glfw.KEY_A and action == glfw.PRESS:
        left = True
    elif key == glfw.KEY_A and action == glfw.RELEASE:
        left = False
    if key == glfw.KEY_D and action == glfw.PRESS:
        right = True
    elif key == glfw.KEY_D and action == glfw.RELEASE:
        right = False
    if key == glfw.KEY_V and action == glfw.PRESS:
        cycle_camera_mode = True
    if key == glfw.KEY_PAGE_UP and action == glfw.PRESS:
        yaw_clockwise = True
    elif key == glfw.KEY_PAGE_UP and action == glfw.RELEASE:
        yaw_clockwise = False
    if key == glfw.KEY_PAGE_DOWN and action == glfw.PRESS:
        yaw_counterclockwise = True
    elif key == glfw.KEY_PAGE_DOWN and action == glfw.RELEASE:
        yaw_counterclockwise = False
    if key == glfw.KEY_1 and action == glfw.PRESS:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    if key == glfw.KEY_2 and action == glfw.PRESS:
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    if key == glfw.KEY_9 and action == glfw.PRESS:
        write_to_gif = not write_to_gif
    if key == glfw.KEY_SPACE and action == glfw.PRESS:
        make_new_ship = True
    if key == glfw.KEY_P and action == glfw.PRESS:
        pause = not pause
    """MENU INPUT PROCESSING"""
    if my_menu.state == 'default':
        if key == glfw.KEY_H and action == glfw.PRESS:
            my_menu.set_state(state="headquarters")
            my_menu.process_state()
    elif my_menu.state == 'headquarters':
        if key == glfw.KEY_I and action == glfw.PRESS:
            headquarters.print_info()
        if key == glfw.KEY_F and action == glfw.PRESS:
            print('| NAME | WALLET | PROFIT |')
            for ship in ships:
                print("| ", ship.name, " | ", ship.wallet, " | ", ship.profit, " | ")
        if key == glfw.KEY_K and action == glfw.PRESS:
            for planet in headquarters.get_planet_lookup().keys():
                print(planet)
        if key == glfw.KEY_N and action == glfw.PRESS:
            pause = True
            price_ship = 100
            choice = ''
            while choice != 'y' and choice != 'n':
                choice = input(f"Buy new ship for {price_ship}? (y/n)")
            if choice == "y":
                if headquarters.wallet < price_ship:
                    print("not enough funds")
                else:
                    print("Buying ship...")
                    ships.append(
                        headquarters.buy_ship(
                            price=price_ship,
                            shader=shader,
                            texture_dictionary=texture_dictionary
                        )
                    )
                    order_list = []
                    orders = ''
                    while orders != 'done':
                        orders = input(f'please enter ship orders (done to finish)')
                        if orders == 'done':
                            continue
                        elif orders == 'move':
                            target = input(f'enter target')
                            order_list.append([orders, headquarters.planet_lookup[target]])
                        elif orders == 'merchant':
                            buy = input(f'enter buy')
                            sell = input('enter sell')
                            order_list.append([orders, {'buy': buy, 'sell': sell}])
                        elif orders == 'repeat':
                            index_repeat = int(input('repeat how many?\n'))
                            print('order_list[-index_repeat:] = ', order_list[-index_repeat:])
                            order_list += order_list[-index_repeat:]
                        elif orders == 'previous':
                            order_list = ships[len(ships)-2].orders
                            orders = 'done'
                        elif orders == 'debug':
                            order_list = debug_ship_orders
                            orders = 'done'

                    ships[len(ships) - 1].minimum_funds = float(input("Enter ship minimum cash"))
                    ships[len(ships) - 1].name = input("Enter ship name")
                    ships[len(ships) - 1].set_orders(orders=order_list)

                    synth = AudioSegment.from_wav("Sounds/synth1.wav")
                    threading.Thread(target=play,
                                     kwargs={'audio_segment': synth},
                                     ).start()
            else:
                print("Not buying ship...  coward")
            pause = False
        if key == glfw.KEY_H and action == glfw.PRESS:
            my_menu.set_state(state="default")
            # my_menu.process_state()


    """GAMEPAD INPUT"""
def gamepad_callback():
    global make_new_ship
    state = glfw.get_gamepad_state(joystick_id=glfw.JOYSTICK_1)
    if not state:
        return
    if state.buttons[glfw.GAMEPAD_BUTTON_A]:
        print('pressed A')
        make_new_ship = True

# do the camera movement, call this function in the main loop
def do_movement(speed=1.0):
    if left:
        cam.process_keyboard("LEFT", 0.05*speed)
    if right:
        cam.process_keyboard("RIGHT", 0.05*speed)
    if forward:
        cam.process_keyboard("FORWARD", 0.05*speed)
    if backward:
        cam.process_keyboard("BACKWARD", 0.05*speed)
    if yaw_clockwise:
        follow_cam.process_keyboard("YAW_CLOCKWISE", 0.05)
    if yaw_counterclockwise:
        follow_cam.process_keyboard("YAW_COUNTERCLOCKWISE", 0.05)


def mouse_look_clb(window, xpos, ypos):
    global first_mouse, lastX, lastY

    if first_mouse:
        lastX = xpos
        lastY = ypos
        first_mouse = False

    xoffset = xpos - lastX
    yoffset = lastY - ypos

    lastX = xpos
    lastY = ypos

    cam.process_mouse_movement(xoffset, yoffset)

def scroll_callback(window, xoffset, yoffset):
    follow_cam.process_mouse_scroll(xoffset, yoffset)


vertex_src = """
# version 330

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texture;
layout(location = 2) in vec3 a_normal;

uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;

out vec2 tex_coords; //texture coordinates
out vec3 normal;
out vec3 frag_pos;


void main()
{
    tex_coords = a_texture;
    frag_pos = vec3(model * vec4(a_position, 1.0));
    normal = mat3(transpose(inverse(model))) * a_normal;
    gl_Position = projection * view * vec4(frag_pos, 1.0);
}
"""

fragment_src = """
#version 330 core
out vec4 frag_color;

struct Material {
    sampler2D diffuse;
    sampler2D specular;
    sampler2D emission;
    float shininess;
}; 

struct DirLight {
    vec3 direction;	
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct PointLight {
    vec3 position; 
    float constant;
    float linear;
    float quadratic;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct SpotLight {
    vec3 position;
    vec3 direction;
    float cut_off;
    float outer_cut_off;
    float constant;
    float linear;
    float quadratic;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;       
};

#define NR_POINT_LIGHTS 2

in vec3 frag_pos;
in vec3 normal;
in vec2 tex_coords;

uniform vec3 view_pos;
uniform DirLight dir_light;
uniform PointLight point_lights[NR_POINT_LIGHTS];
uniform SpotLight spot_light;
uniform Material material;

// function prototypes
vec3 CalcDirLight(DirLight light, vec3 normal, vec3 view_dir);
vec3 CalcPointLight(PointLight light, vec3 normal, vec3 frag_pos, vec3 view_dir);
vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 frag_pos, vec3 view_dir);

void main()
{    
    // properties
    vec3 norm = normalize(normal);
    vec3 view_dir = normalize(view_pos - frag_pos);

    // == =====================================================
    // Our lighting is set up in 3 phases: directional, point lights and an optional flashlight
    // For each phase, a calculate function is defined that calculates the corresponding color
    // per lamp. In the main() function we take all the calculated colors and sum them up for
    // this fragment's final color.
    // == =====================================================
    vec3 result;
    // phase 1: directional lighting
    result = CalcDirLight(dir_light, norm, view_dir);
    // phase 2: point lights
    for(int i = 0; i < NR_POINT_LIGHTS; i++)
        result += CalcPointLight(point_lights[i], norm, frag_pos, view_dir);    
    // phase 3: spotlight
    result += CalcSpotLight(spot_light, norm, frag_pos, view_dir);    
    // emission
    vec3 emission = texture(material.emission, tex_coords).rgb;
    result = result + emission;
    frag_color = vec4(result, 1.0);
}

// calculates the color when using a directional light.
vec3 CalcDirLight(DirLight light, vec3 normal, vec3 view_dir)
{
    vec3 light_dir = normalize(-light.direction);

    // diffuse shading
    float diff = max(dot(normal, light_dir), 0.0);
    // specular shading
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), material.shininess);
    // combine results
    vec3 ambient = light.ambient * vec3(texture(material.diffuse, tex_coords));
    vec3 diffuse = light.diffuse * diff * vec3(texture(material.diffuse, tex_coords));
    vec3 specular = light.specular * spec * vec3(texture(material.specular, tex_coords));
    return (ambient + diffuse + specular);
}

// calculates the color when using a point light.
vec3 CalcPointLight(PointLight light, vec3 normal, vec3 frag_pos, vec3 view_dir)
{
    vec3 light_dir = normalize(light.position - frag_pos);
    // diffuse shading
    float diff = max(dot(normal, light_dir), 0.0);
    // specular shading
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), material.shininess);
    // attenuation
    float distance = length(light.position - frag_pos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));    
    // combine results
    vec3 ambient = light.ambient * vec3(texture(material.diffuse, tex_coords));
    vec3 diffuse = light.diffuse * diff * vec3(texture(material.diffuse, tex_coords));
    vec3 specular = light.specular * spec * vec3(texture(material.specular, tex_coords));
    ambient *= attenuation;
    diffuse *= attenuation;
    specular *= attenuation;
    return (ambient + diffuse + specular);
}

// calculates the color when using a spot light.
vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 frag_pos, vec3 view_dir)
{
    vec3 light_dir = normalize(light.position - frag_pos);
    //view_dir = vec3(0.0, -1.0, 0.0); //tp

    // diffuse shading
    float diff = max(dot(normal, light_dir), 0.0);

    // specular shading
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), material.shininess);

    // attenuation
    float distance = length(light.position - frag_pos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));    
    // spotlight intensity
    float theta = dot(light_dir, normalize(-light.direction)); 
    float epsilon = light.cut_off - light.outer_cut_off;
    float intensity = clamp((theta - light.outer_cut_off) / epsilon, 0.0, 1.0);
    // combine results
    vec3 ambient = light.ambient * vec3(texture(material.diffuse, tex_coords));
    vec3 diffuse = light.diffuse * diff * vec3(texture(material.diffuse, tex_coords));
    vec3 specular = light.specular * spec * vec3(texture(material.specular, tex_coords));
    ambient *= attenuation * intensity;
    diffuse *= attenuation * intensity;
    specular *= attenuation * intensity;
    return (ambient + diffuse + specular);
}
"""

# the window resize callback function
def window_resize_clb(window, width, height):
    glViewport(0, 0, width, height)
    projection = pyrr.matrix44.create_perspective_projection_matrix(45, width / height, 0.1, 2000)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)


# initializing glfw library
if not glfw.init():
    raise Exception("glfw can not be initialized!")

# creating the window
window = glfw.create_window(WIDTH, HEIGHT, "Spaceship Generator", None, None)

# check if window was created
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")

# set window's position
glfw.set_window_pos(window, 400, 200)

# set the callback function for window resize
glfw.set_window_size_callback(window, window_resize_clb)
# set the mouse position callback
glfw.set_cursor_pos_callback(window, mouse_look_clb)
# set the keyboard input callback
glfw.set_key_callback(window, key_input_clb)
#set the scroll-wheel input callback
glfw.set_scroll_callback(window, scroll_callback)
# capture the mouse cursor
glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

# make the context current
glfw.make_context_current(window)

shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))


glEnable(GL_CULL_FACE)

#uncomment to see that cull is working by culling front faces rather than back
# glCullFace(GL_BACK)
# glFrontFace(GL_CW)

glUseProgram(shader)

# set the texture unit (integer) of each sampler2D
glUniform1i(glGetUniformLocation(shader, "material.diffuse"), 0)
glUniform1i(glGetUniformLocation(shader, "material.specular"), 1)
glUniform1i(glGetUniformLocation(shader, "material.emission"), 2)

glClearColor(0, 0.1, 0.1, 1)
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

projection_2 = pyrr.matrix44.create_perspective_projection_matrix(45, WIDTH / HEIGHT, 0.1, 2000)

model_loc = glGetUniformLocation(shader, "model")
proj_loc = glGetUniformLocation(shader, "projection")
view_loc = glGetUniformLocation(shader, "view")

view_pos_loc = glGetUniformLocation(shader, "view_pos")  # cam pos for specular lighting

glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection_2)

# create the light cube
my_plc = plc.PointLightCube(pos=[0.0, 0.0, 0.0], ambient=[0.5]*3, diffuse=[0.9]*3,
                            specular=[1.0]*3)
debug_plcs = []
debug_plcs.append(plc.PointLightCube(
    pos=[0.0, 5.0, 5.0],
    ambient=[0.1]*3,
    diffuse=[0.9]*3,
    specular=[0.9]*3
))
debug_plcs.append(plc.PointLightCube(
    pos=[0.0, 5.0, 10.0],
    ambient=[0.0, 0.0, 1.0],
    diffuse=[0.0, 0.0, 1.0],
    specular=[0.0, 0.0, 1.0],
))
debug_plcs.append(plc.PointLightCube(
    pos=[0.0, 5.0, 0.0],
    ambient=[1.0, 0.0, 0.0],
    diffuse=[1.0, 0.0, 0.0],
    specular=[1.0, 0.0, 0.0],
))
debug_plcs.append(plc.PointLightCube(
    pos=[5.0, 5.0, 5.0],
    ambient=[0.0, 1.0, 0.0],
    diffuse=[0.0, 1.0, 0.0],
    specular=[0.0, 1.0, 0.0],
))
debug_plcs.append(plc.PointLightCube(
    pos=[-5.0, 5.0, 5.0],
    ambient=[1.0, 1.0, 0.0],
    diffuse=[1.0, 1.0, 0.0],
    specular=[1.0, 1.0, 0.0],
))
# nighty night mode
# debug_plcs.append(plc.PointLightCube(
#     pos=[0.0, 5.0, 5.0],
#     ambient=[0.0]*3,
#     diffuse=[0.0]*3,
#     specular=[0.0]*3
# ))
# full lights mode
# debug_plcs.append(plc.PointLightCube(
#     pos=[0.0, 5.0, 5.0],
#     ambient=[0.95]*3,
#     diffuse=[0.95]*3,
#     specular=[0.95]*3
# ))

#seed random
seed = 1999
r.seed(seed)

"""direction light settings"""
direction_diffuse = [0.0]*3
direction_ambient = [0.0]*3
direction_specular = [0.0]*3

# direction light pass to shader
glUniform3fv(glGetUniformLocation(shader, "dir_light.position"), 1, [0, 20, 10])
glUniform3fv(glGetUniformLocation(shader, "dir_light.diffuse"), 1, direction_diffuse)
glUniform3fv(glGetUniformLocation(shader, "dir_light.ambient"), 1, direction_ambient)
glUniform3fv(glGetUniformLocation(shader, "dir_light.specular"), 1, direction_specular)


"""Tile Sets"""
#simplest possible tileset: cross, straight(x2), and blank
tile_textures = glGenTextures(13)
load_texture("Textures/penguin_atlas_emission.png", tile_textures[0])
load_texture("Textures/penguin_atlas_emission.png", tile_textures[1])
load_texture("Textures/beige_atlas_diffuse.png", tile_textures[2])
load_texture("Textures/debug_quad_red.png", tile_textures[3])
load_texture("Textures/penguin_atlas_specular.png", tile_textures[4])
load_texture("Textures/penguin_atlas_specular.png", tile_textures[5])
load_texture("Textures/debug_texture_atlas.png", tile_textures[6])
load_texture("Textures/spaceship_texture_atlas_1.png", tile_textures[7])
load_texture("Textures/spaceship_texture_atlas_1_specular.png", tile_textures[8])
load_texture("Textures/spaceship_texture_atlas_1_emission.png", tile_textures[9])
load_texture("Textures/debug_texture_atlas_2.png", tile_textures[10])
load_texture("Textures/debug_texture_atlas_2_emission.png", tile_textures[11])
load_texture("Textures/debug_texture_atlas_2_specular.png", tile_textures[12])


texture_dictionary = {
    "pink": tile_textures[3],
    "penguin_diffuse": tile_textures[2],
    "penguin_emission": tile_textures[1],
    "penguin_specular": tile_textures[4],
    "spaceship_diffuse": tile_textures[7],
    "spaceship_specular": tile_textures[8],
    "spaceship_emission": tile_textures[9],
    "atlas_debug_diffuse": tile_textures[10],
    "atlas_debug_emission": tile_textures[11],
    "atlas_debug_specular": tile_textures[12],
}

headquarters_model = ProceduralMesh.CubeMeshStatic(
    shader=shader,
    diffuse=texture_dictionary['pink'],
    specular=texture_dictionary['pink'],
    dimensions=[3.0, 3.0, 3.0],
    position=[20.0, 10.0, -20.0]
)
headquarters = Headquarters(model=headquarters_model)

testing_model = primatives.Prism(
    shader=shader,
    diffuse=texture_dictionary['pink'],
    specular=texture_dictionary['pink'],
    dimensions=[2.0, 2.0],
    position=[25.0, 10, 25],
    sides=4,
)
testing_model_b = primatives.Prism(
    shader=shader,
    diffuse=texture_dictionary['pink'],
    specular=texture_dictionary['pink'],
    dimensions=[2.0, 2.0],
    position=[-25.0, -10, -25],
    sides=8,
)
cargo_a = defaultdict(int)
cargo_a['lemonanas'] = 10000
cargo_a['tritium'] = 200
prices_a = defaultdict(int)
prices_a['lemonanas'] = 1.2
prices_a['tritium'] = 120

cargo_b = defaultdict(int)
cargo_b['lemonanas'] = 10000
cargo_b['tritium'] = 200
prices_b = defaultdict(int)
prices_b['lemonanas'] = 1.7
prices_b['tritium'] = 100


beacon = SpaceShip.Beacon(
    position=testing_model.position,
    model=testing_model,
    cargo=cargo_a,
    prices=prices_a,
    name="avor",
)
beacon_b = SpaceShip.Beacon(
    position=testing_model_b.position,
    model=testing_model_b,
    cargo=cargo_b,
    prices=prices_b,
    name="skalga",
)
headquarters.update_planet_lookup(headquarters)
headquarters.update_planet_lookup(beacon_b)
headquarters.update_planet_lookup(beacon)

debug_ship_orders = [
        ['move', beacon],
        ['merchant', {'buy': 'lemonanas', 'sell': 'tritium'}],
        ['move', beacon_b],
        ['merchant', {'buy': 'tritium', 'sell': 'lemonanas'}],
        ['move', headquarters]
    ]

spaceship_parameters = {
    'number_of_sides': 4,
    'number_of_segments': 2,
    # 'transform_x': 0.25 + random()*1.25,
    'transform_x': 0.4,
    'transform_z': 0.4,
}



ships = []
num_ships = 0
for ship in range(num_ships):

    spaceship = primatives.Spaceship(
        shader=shader,
        # diffuse=texture_dictionary['atlas_debug_diffuse'],
        # specular=texture_dictionary['atlas_debug_specular'],
        # emission=texture_dictionary['atlas_debug_emission'],
        # diffuse=texture_dictionary['spaceship_diffuse'],
        # specular=texture_dictionary['spaceship_specular'],
        # emission=texture_dictionary['spaceship_emission'],
        diffuse=texture_dictionary["penguin_diffuse"],
        emission=texture_dictionary["penguin_emission"],
        specular=texture_dictionary["penguin_specular"],
        dimensions=[5.0, 5.0],
        position=[(random()-0.5)*100, (random()-0.5)*50, (random()-0.5)*50],
        rotation_magnitude=[0.0, 0.0, -m.pi*0.5],
        number_of_sides=spaceship_parameters['number_of_sides'],
        # number_of_segments=spaceship_parameters['number_of_segments'],
        number_of_segments=math.floor(random() * 7 + 3),
        transform_x=spaceship_parameters['transform_x'],
        transform_z=spaceship_parameters['transform_z'],
        length_of_segment=5.0,
        radius=3.0,
        scale=random()*0.2 + 0.1
    )
    ship_current = SpaceShip.Spaceship(model=spaceship, wallet=50)
    ship_current.set_velocity(velocity=[-1.5, 0.0, 0.0])
    ships.append(ship_current)

    ship_current.set_orders(orders=[
        ['move', beacon],
        ['merchant', {'buy': 'lemonanas', 'sell': 'tritium'}],
        ['move', beacon_b],
        ['merchant', {'buy': 'tritium', 'sell': 'lemonanas'}],
        ['move', headquarters]
    ]
    )


bezier_cube = primatives.NPrismBezierCut(
    shader=shader,
    diffuse=texture_dictionary["atlas_debug_diffuse"],
    specular=texture_dictionary["penguin_diffuse"],
    shininess=32.0,
    dimensions=[2.0],
    position=[0.0, 5.0, -5.0],
    rotation_axis=glm.vec3([0.0, 1.0, 0.0]),
    rotation_magnitude=[0.0, 0.0, m.pi],
    scale=glm.vec3([5.0]*3),
    sides=5
)


testing_model_polygon = primatives.Polygon(
    shader=shader,
    diffuse=texture_dictionary['penguin_diffuse'],
    specular=texture_dictionary['penguin_diffuse'],
    dimensions=[20.0, 2.0],
    position=[5.0, 0.0, 0.0],
    sides=4,
)

testing_primative_mesh_emission = primatives.PrimativeMeshEmission(
    shader=shader,
    diffuse=texture_dictionary['atlas_debug_diffuse'],
    specular=texture_dictionary['atlas_debug_specular'],
    emission=texture_dictionary['atlas_debug_emission'],
    dimensions=[10.0, 10.0],
    position=[20.0, 10.0, 0.0],
)



my_fps = FPSCounter.FPSCounter(frame_interval=300.0, mute=True)


background_music_1 = AudioSegment.from_wav("Sounds/PortalToUnderworld_1.wav")
background_music_2 = AudioSegment.from_wav("Sounds/PinkBloom.wav")
background_music_3 = AudioSegment.from_wav("Sounds/DavidKBD - Pink Bloom Pack - 03 - To the Unknown.wav")
background_music_4 = AudioSegment.from_wav("Sounds/DavidKBD - Pink Bloom Pack - 04 - Valley of Spirits.wav")
synth = AudioSegment.from_wav("Sounds/synth1.wav")

background_playlist = background_music_4 + background_music_1 + background_music_3 + background_music_2
background_playlist *= 5
#-50 db is very quiet
background_playlist -= 100
"""Sound Startup"""
thread_jukebox = threading.Thread(
            target=play,
            kwargs={'audio_segment': background_playlist},
            daemon=True,
        )
thread_jukebox.start()

#todo: make this a class or method at least
#MAP GENERATION: generate points of starts
map_x = 5
map_z = 5
distance_star = 5.0
galaxy_map = {}
disk_shaped = True
center_galaxy = glm.vec2(map_x/2, map_z/2)
radius_galaxy = (map_x + map_z) / 4
# radius_galaxy = 4
height_max_star = 5
probability_prune = 0.5
for i in range(map_x):
    for j in range(map_z):
        random_offset_x = random()*4*distance_star - 3*distance_star
        random_offset_z = random()*4*distance_star - 3*distance_star
        if disk_shaped:
            if glm.distance(glm.vec2(i, j), center_galaxy) < radius_galaxy and random() > probability_prune:
                galaxy_map[(i, j)] = [
                    2 * distance_star + 4 * distance_star * i + random_offset_x,
                    random()*height_max_star,
                    2 * distance_star + 4 * distance_star * j + random_offset_z
                ]


stars = []
planet_number = 1
for coordinates, position in galaxy_map.items():
    print(position)
    star_model = primatives.Prism(
            shader=shader,
            diffuse=texture_dictionary['pink'],
            specular=texture_dictionary['pink'],
            dimensions=[6.0, 6.0],
            position=position,
            sides=3,
        )
    star = SpaceShip.Beacon(
        position=position,
        model=star_model,
        cargo=cargo_a,
        prices=prices_a,
        name=str(planet_number),
    )
    planet_number += 1
    headquarters.update_planet_lookup(star)
    stars.append(star)


while not glfw.window_should_close(window):
    glfw.poll_events()
    gamepad_callback()
    do_movement(speed=10)
    my_fps.update()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    ct = glfw.get_time()

    view = cam.get_view_matrix()
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

    for star in stars:
        star.draw(view=view)





    "process ships"
    for ship in ships:
        if not pause:
            ship.process_states(delta_time=0.1)
            # beacon.set_position(position=[m.cos(glfw.get_time() * 0.5) * 30, m.cos(glfw.get_time() * 0.25) * 10.0,
            #                               m.sin(glfw.get_time() * 0.5) * 30])
            # beacon_b.set_position(position=[m.cos(glfw.get_time() * 0.25) * -50, m.sin(glfw.get_time() * 0.25) * 10.0,
            #                               m.sin(glfw.get_time() * 0.5) * 30])
        ship.draw(view=view)
    headquarters.draw(view=view)

    """draw new ship"""
    if make_new_ship:
        make_new_ship = False
        seed += 1
        spaceship = primatives.Spaceship(
            shader=shader,
            diffuse=texture_dictionary["penguin_diffuse"],
            specular=texture_dictionary["penguin_specular"],
            emission=texture_dictionary["penguin_emission"],
            dimensions=[5.0, 5.0],
            position=ships[0].model.position,
            rotation_magnitude=ships[0].model.rotation_magnitude,
            rotation_axis=glm.vec3([0.0, 0.0, 1.0]),
            number_of_sides=spaceship_parameters['number_of_sides'],
            number_of_segments=spaceship_parameters['number_of_segments'],
            transform_x=spaceship_parameters['transform_x'],
            transform_z=spaceship_parameters['transform_z'],
            length_of_segment=5.0,
            radius=3.0,
        )
        ships[0].model = spaceship
        # ships[0].set_target(target=beacon)
        # ships[0].set_orders(orders=)
    # if not pause:
    #     pass
    #     # beacon.set_position(position=[m.cos(glfw.get_time()*0.5)*30, m.cos(glfw.get_time()*0.25)*10.0, m.sin(glfw.get_time()*0.5)*30])

    beacon.draw(view=view)
    beacon_b.draw(view=view)


    # draw the point light cube
    # my_plc.draw(view)
    for light in debug_plcs:
        light.draw(view)
    glUseProgram(shader)

    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

    projection = pyrr.matrix44.create_perspective_projection_matrix(45, WIDTH / HEIGHT, 0.1, 2050)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

    # pass cam position for specular light
    glUniform3fv(view_pos_loc, 1, list(cam.camera_pos))

    point_light_position_loc = glGetUniformLocation(shader, "point_light.position")


    glUniform1f(glGetUniformLocation(shader, "material.shininess"), 32.0)

    # light color components to shader
    # glUniform3fv(glGetUniformLocation(shader, "point_lights[0].position"), 1, my_plc.get_pos())
    glUniform3fv(glGetUniformLocation(shader, "point_lights[0].position"), 1, [0.0, 0.0, 0.0])
    glUniform3fv(glGetUniformLocation(shader, "point_lights[0].diffuse"), 1, my_plc.get_diffuse())
    glUniform3fv(glGetUniformLocation(shader, "point_lights[0].ambient"), 1, my_plc.get_ambient())
    glUniform3fv(glGetUniformLocation(shader, "point_lights[0].specular"), 1, my_plc.get_specular())
    glUniform1f(glGetUniformLocation(shader, "point_lights[0].constant"), my_plc.get_constant())
    glUniform1f(glGetUniformLocation(shader, "point_lights[0].linear"), my_plc.get_linear())
    glUniform1f(glGetUniformLocation(shader, "point_lights[0].quadratic"), my_plc.get_quadratic())

    #second light for mesh viewing
    glUniform3fv(glGetUniformLocation(shader, "point_lights[1].position"), 1, debug_plcs[0].get_pos())
    glUniform3fv(glGetUniformLocation(shader, "point_lights[1].diffuse"), 1, debug_plcs[0].get_diffuse())
    glUniform3fv(glGetUniformLocation(shader, "point_lights[1].ambient"), 1, debug_plcs[0].get_ambient())
    glUniform3fv(glGetUniformLocation(shader, "point_lights[1].specular"), 1, debug_plcs[0].get_specular())
    glUniform1f(glGetUniformLocation(shader, "point_lights[1].constant"), debug_plcs[0].get_constant())
    glUniform1f(glGetUniformLocation(shader, "point_lights[1].linear"), debug_plcs[0].get_linear())
    glUniform1f(glGetUniformLocation(shader, "point_lights[1].quadratic"), debug_plcs[0].get_quadratic())

    # spot light
    glUniform3fv(glGetUniformLocation(shader, "spot_light.position"), 1, list(cam.camera_pos))
    glUniform3fv(glGetUniformLocation(shader, "spot_light.direction"), 1, list(cam.camera_front))
    glUniform3fv(glGetUniformLocation(shader, "spot_light.diffuse"), 1, [0.0]*3)
    glUniform3fv(glGetUniformLocation(shader, "spot_light.ambient"), 1, [0.0]*3)
    glUniform3fv(glGetUniformLocation(shader, "spot_light.specular"), 1, [0.0]*3)
    glUniform1f(glGetUniformLocation(shader, "spot_light.cut_off"), glm.cos(glm.radians(12.5)))
    glUniform1f(glGetUniformLocation(shader, "spot_light.outer_cut_off"), glm.cos(glm.radians(45.0)))
    glUniform1f(glGetUniformLocation(shader, "spot_light.constant"), 1.0)
    glUniform1f(glGetUniformLocation(shader, "spot_light.linear"), 0.00003)
    glUniform1f(glGetUniformLocation(shader, "spot_light.quadratic"), 0.00007)

    if write_to_gif:
        window_as_numpy_arr = np.uint8(glReadPixels(0,0,800,800,GL_RGB, GL_FLOAT)*255.0)
        window_as_numpy_arr = np.flip(window_as_numpy_arr,0)
        window_as_PIL_image = Image.fromarray(window_as_numpy_arr)
        images.append(window_as_PIL_image)

    glfw.swap_buffers(window)

if write_to_gif:
    images[0].save(
            'spaceship.gif',
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=20,
            loop=0
        )
glfw.terminate()
