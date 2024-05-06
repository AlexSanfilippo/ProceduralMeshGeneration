"""
6th April, 2024
last update: 6th April 2024
Displaying Proc-Gen Meshes for testing/developement

-[V]Make this file
-[V]Add code to generate and draw meshes in a meshes list which is drawn
-[V]remove code related to ship trading game
-[V]fix texture coordinate bug in bevel_cut
    -[V]extruded face bug
    -[V]side faces bug
        -requires doing glTexCoord4f(s,t,0,q) instead of glTexCoord2f(s,t)
            -where?
            -where s,t are our texture coords as they are
            -must calculate q
-[V]Change texture filtering to Nearest
-[V]Add UI
    -[V]Display textured quad on screen
    -[V]click textured quad-change quad color or something

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
from Camera import Camera, FollowCamera, SimulationCamera
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

follow_cam = SimulationCamera(camera_pos=[6.0, 247.0, 242.0])
cam = Camera(camera_pos=[0.0, 20.0, 20.0])
use_follow_cam = False



WIDTH, HEIGHT = 1280, 720
lastX, lastY = WIDTH / 2, HEIGHT / 2
first_mouse = True
left, right, forward, backward, make_new_surface = False, False, False, False, False
player_left, player_right, player_forward, player_backward = False, False, False, False
yaw_counterclockwise, yaw_clockwise = False, False
up, down = False, False
write_to_gif = False
make_new_ship = False
pause = False
ship_texture_cycle_id = 0


"""Text MENU TESTING"""
my_menu = Menu()
# the keyboard input callback
def key_input_clb(window, key, scancode, action, mode):
    global left, right, forward, backward, make_new_surface, player_left, player_right, player_forward, \
        player_backward, yaw_counterclockwise, yaw_clockwise, write_to_gif, make_new_ship,\
        pause, ship_texture_cycle_id, up, down

    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)
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
    if key == glfw.KEY_Q and action == glfw.PRESS:
        yaw_clockwise = True
    elif key == glfw.KEY_Q and action == glfw.RELEASE:
        yaw_clockwise = False
    if key == glfw.KEY_E and action == glfw.PRESS:
        yaw_counterclockwise = True
    elif key == glfw.KEY_E and action == glfw.RELEASE:
        yaw_counterclockwise = False
    if key == glfw.KEY_TAB and action == glfw.PRESS:
        up = True
    elif key == glfw.KEY_TAB and action == glfw.RELEASE:
        up = False
    if key == glfw.KEY_LEFT_SHIFT and action == glfw.PRESS:
        down = True
    elif key == glfw.KEY_LEFT_SHIFT and action == glfw.RELEASE:
        down = False

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
    if key == glfw.KEY_T and action == glfw.PRESS:
        cycle_ship_texture()
    if key == glfw.KEY_V and action == glfw.PRESS:
        switch_camera_mode()


def cycle_ship_texture():
    global ship_texture_cycle_id
    if ship_texture_cycle_id < 4:
        ship_texture_cycle_id += 1
    else:
        ship_texture_cycle_id = 0
    if ship_texture_cycle_id == 0:
        spaceship_parameters['diffuse'] = texture_dictionary['penguin_diffuse']
        spaceship_parameters['specular'] = texture_dictionary['penguin_specular']
        spaceship_parameters['emission'] = texture_dictionary['penguin_emission']
        update_spaceship_texture()
    elif ship_texture_cycle_id == 1:
        spaceship_parameters['diffuse'] = texture_dictionary['atlas_debug_diffuse']
        spaceship_parameters['specular'] = texture_dictionary['atlas_debug_specular']
        spaceship_parameters['emission'] = texture_dictionary['atlas_debug_emission']
        update_spaceship_texture()
    elif ship_texture_cycle_id == 2:
        spaceship_parameters['diffuse'] = texture_dictionary['whoa_diffuse']
        spaceship_parameters['specular'] = texture_dictionary['whoa_specular']
        spaceship_parameters['emission'] = texture_dictionary['atlas_debug_emission']
        update_spaceship_texture()
    elif ship_texture_cycle_id == 3:
        spaceship_parameters['diffuse'] = texture_dictionary['spaceship_diffuse']
        spaceship_parameters['specular'] = texture_dictionary['spaceship_specular']
        spaceship_parameters['emission'] = texture_dictionary['spaceship_emission']
        update_spaceship_texture()
    elif ship_texture_cycle_id == 4:
        spaceship_parameters['diffuse'] = texture_dictionary['ship_a_diffuse']
        spaceship_parameters['specular'] = texture_dictionary['ship_a_specular']
        spaceship_parameters['emission'] = texture_dictionary['ship_a_emission']
        update_spaceship_texture()


def mouse_button_callback(window, button, action, mods):
    global make_new_ship
    left_click = button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS
    right_click = button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS
    if button == left_click:
       mpos = glfw.get_cursor_pos(window)
    test_gui.button_update(position_mouse=glfw.get_cursor_pos(window), left_click=left_click, right_click=right_click)

def update_spaceship_texture():
    ships[0].model.set_diffuse(spaceship_parameters['diffuse'])
    ships[0].model.set_specular(spaceship_parameters['specular'])
    ships[0].model.set_emission(spaceship_parameters['emission'])


def do_movement(speed=1.0):
    """
    do the camera movement, call this function in the main loop
    :param speed:
    :return:
    """
    if left:
        active_camera.process_keyboard("LEFT", 0.05*speed)
    if right:
        active_camera.process_keyboard("RIGHT", 0.05*speed)
    if forward:
        active_camera.process_keyboard("FORWARD", 0.05*speed)
    if backward:
        active_camera.process_keyboard("BACKWARD", 0.05*speed)
    if yaw_clockwise:
        follow_cam.process_keyboard("YAW_CLOCKWISE", 0.05)
    if yaw_counterclockwise:
        follow_cam.process_keyboard("YAW_COUNTERCLOCKWISE", 0.05)
    if up:
        active_camera.process_keyboard("UP", 0.05)
    if down:
        active_camera.process_keyboard("DOWN", 0.05)




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
    
    /*
    tex_coords = a_texture;
    float q_coord = 0.9;    
    if (tex_coords == vec2(1,1) || tex_coords == vec2(0,0)){
        tex_coords = tex_coords / q_coord;
        //tex_coords.x = tex_coords.x / q_coord;
    }
    */
    
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
vec3 CalcDirLight(DirLight light, vec3 normal, vec3 view_dir, vec2 tex_coords);
vec3 CalcPointLight(PointLight light, vec3 normal, vec3 frag_pos, vec3 view_dir, vec2 tex_coords);
vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 frag_pos, vec3 view_dir);

void main()
{    
    vec2 tex_coords_alternate =  tex_coords;
    
    /*
    float q_coord = 0.1f;
    if (tex_coords_alternate.x > 0.9f && tex_coords_alternate.y > 0.9f){
        tex_coords_alternate = tex_coords_alternate * q_coord;
    }
    if (tex_coords_alternate == vec2(0.f, 1.f)){
        tex_coords_alternate = tex_coords_alternate * q_coord;
    }
    */
        
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
    result = CalcDirLight(dir_light, norm, view_dir, tex_coords_alternate);
    // phase 2: point lights
    for(int i = 0; i < NR_POINT_LIGHTS; i++)
        result += CalcPointLight(point_lights[i], norm, frag_pos, view_dir, tex_coords_alternate);    
    // phase 3: spotlight
    //result += CalcSpotLight(spot_light, norm, frag_pos, view_dir);    
    // emission
    vec3 emission = texture(material.emission, tex_coords_alternate).rgb;
    result = result + emission;
    frag_color = vec4(result, 1.0);
}

// calculates the color when using a directional light.
vec3 CalcDirLight(DirLight light, vec3 normal, vec3 view_dir, vec2 tex_coords)
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
vec3 CalcPointLight(PointLight light, vec3 normal, vec3 frag_pos, vec3 view_dir, vec2 tex_coords)
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
#set the mouse callback
glfw.set_mouse_button_callback(window, mouse_button_callback)
# capture the mouse cursor
# glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
# glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_CAPTURED)

# make the context current
glfw.make_context_current(window)

def switch_camera_mode():
    global use_follow_cam, active_camera
    use_follow_cam = not use_follow_cam
    if use_follow_cam:
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL) #glfw.CURSOR_CAPTURED,
        active_camera = follow_cam
    else:
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        active_camera = cam
switch_camera_mode()

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


"""direction light settings"""
direction_diffuse = [0.0]*3
direction_ambient = [0.0]*3
direction_specular = [0.0]*3
# create the light cube
my_plc = plc.PointLightCube(pos=[0.0, 20.0, 0.0], ambient=[0.2]*3, diffuse=[0.9]*3,
                            specular=[0.0]*3, linear=0.0014, quadratic=.000007)
debug_plcs = [my_plc]
move_whole_compass_z = -80
debug_plcs.append(plc.PointLightCube(
    pos=[0.0, 10.0, move_whole_compass_z+5.0],
    ambient=[0.1]*3,
    diffuse=[0.1]*3,
    specular=[0.1]*3
))
debug_plcs.append(plc.PointLightCube(
    pos=[0.0, 5.0, move_whole_compass_z+10.0],
    ambient=[0.0, 0.0, 1.0],
    diffuse=[0.0, 0.0, 1.0],
    specular=[0.0, 0.0, 1.0],
))
debug_plcs.append(plc.PointLightCube(
    pos=[0.0, 5.0, move_whole_compass_z+0.0],
    ambient=[1.0, 0.0, 0.0],
    diffuse=[1.0, 0.0, 0.0],
    specular=[1.0, 0.0, 0.0],
))
debug_plcs.append(plc.PointLightCube(
    pos=[5.0, 5.0, move_whole_compass_z+5.0],
    ambient=[0.0, 1.0, 0.0],
    diffuse=[0.0, 1.0, 0.0],
    specular=[0.0, 1.0, 0.0],
))
debug_plcs.append(plc.PointLightCube(
    pos=[-5.0, 5.0, move_whole_compass_z+5.0],
    ambient=[1.0, 1.0, 0.0],
    diffuse=[1.0, 1.0, 0.0],
    specular=[1.0, 1.0, 0.0],
))

seed = 1999
r.seed(seed)



# direction light pass to shader
glUniform3fv(glGetUniformLocation(shader, "dir_light.position"), 1, [0, 20, 10])
glUniform3fv(glGetUniformLocation(shader, "dir_light.diffuse"), 1, direction_diffuse)
glUniform3fv(glGetUniformLocation(shader, "dir_light.ambient"), 1, direction_ambient)
glUniform3fv(glGetUniformLocation(shader, "dir_light.specular"), 1, direction_specular)


"""Tile Sets"""
#simplest possible tileset: cross, straight(x2), and blank
tile_textures = glGenTextures(21)
load_texture("Textures/beige_atlas_diffuse.png", tile_textures[0])
load_texture("Textures/whoa_atlas_specular.png", tile_textures[1])
load_texture("Textures/penguin_atlas_emission.png", tile_textures[2])
load_texture("Textures/debug_quad_red.png", tile_textures[3])
load_texture("Textures/penguin_atlas_specular.png", tile_textures[4])
load_texture("Textures/penguin_atlas_specular.png", tile_textures[5])
load_texture("Textures/debug_texture_atlas.png", tile_textures[6])
load_texture("Textures/spaceship_texture_atlas_1.png", tile_textures[7])
load_texture("Textures/spaceship_texture_atlas_1_specular.png", tile_textures[8])
load_texture("Textures/spaceship_texture_atlas_1_emission.png", tile_textures[9])
load_texture("Textures/debug_diffuse_coordinates.png", tile_textures[10])
load_texture("Textures/penguin_atlas_emission.png", tile_textures[11])
load_texture("Textures/penguin_atlas_specular.png", tile_textures[12])
load_texture("Textures/whoa_atlas_diffuse.png", tile_textures[13])
load_texture("Textures/whoa_atlas_specular.png", tile_textures[14])
load_texture("Textures/penguin_atlas_emission.png", tile_textures[15])
load_texture("Fonts/my_font.png", tile_textures[16])
load_texture("Textures/button_atlas_workshop.png", tile_textures[17])

load_texture("Textures/ship_a_diffuse.png", tile_textures[18])
load_texture("Textures/ship_a_specular.png", tile_textures[19])
load_texture("Textures/ship_a_emission.png", tile_textures[20])


texture_dictionary = {
    "pink": tile_textures[3],
    "penguin_diffuse": tile_textures[0],
    "penguin_specular": tile_textures[1],
    "penguin_emission": tile_textures[2],
    "spaceship_diffuse": tile_textures[7],
    "spaceship_specular": tile_textures[8],
    "spaceship_emission": tile_textures[9],
    "atlas_debug_diffuse": tile_textures[10],
    "atlas_debug_emission": tile_textures[11],
    "atlas_debug_specular": tile_textures[12],
    "whoa_diffuse": tile_textures[13],
    "whoa_specular": tile_textures[14],
    "whoa_emission": tile_textures[15],
    "font_atlas": tile_textures[16],
    "button_atlas": tile_textures[17],
    "ship_a_diffuse": tile_textures[18],
    "ship_a_specular": tile_textures[19],
    "ship_a_emission": tile_textures[20],
}


debug_ship_orders = [
        ['stuck']
]

spaceship_parameters = {
    'number_of_sides': 8,
    'number_of_segments': 10,
    'transform_x': 1.0,
    'transform_z': 2.0,
    'scale': 3.3,
    'diffuse': texture_dictionary['ship_a_diffuse'],
    'specular': texture_dictionary['ship_a_specular'],
    'emission': texture_dictionary['ship_a_emission'],
    # 'diffuse': texture_dictionary['spaceship_diffuse'],
    # 'specular': texture_dictionary['spaceship_specular'],
    # 'emission': texture_dictionary['spaceship_emission'],
    # 'diffuse': texture_dictionary['whoa_diffuse'],
    # 'specular': texture_dictionary['whoa_specular'],
    # 'emission': texture_dictionary['whoa_emission'],
    'position': [0.0, 0.0, 0.0],
    'length_of_segment': 10,
}


"""Texture Coordinate Debugging Models"""
shapes = []
for sides in list(range(3, 13)):
    shapes.append(primatives.Polygon(
        shader=shader,
        diffuse=spaceship_parameters['diffuse'],
        specular=spaceship_parameters['specular'],
        dimensions=[5.0, 5.0],
        position=[0.0, 0.0, sides*10],
        rotation_magnitude=[0.0, 0.0, -m.pi * 0.5],
        scale=spaceship_parameters['scale'],
        sides=sides,
        transform_x=2.0,
        transform_z=1.0,
    ))

"""Testing trapezoid texturing algorithm"""
shapes.append(
    primatives.PolygonIrregular(
        shader=shader,
        diffuse=spaceship_parameters['diffuse'],
        specular=spaceship_parameters['specular'],
        dimensions=[5.0, 5.0],
        position=[10.0, 0.0, 40],
        rotation_magnitude=[0.0, 0.0, -m.pi * 0.5],
        scale=spaceship_parameters['scale'],
        sides=4,
        transform_x=1.0,
        transform_z=1.0,
        radii=[2.0, 1.0, 1.0, 2.0]
    )
)

ships = []
num_ships = 1
for ship in range(num_ships):
    spaceship = primatives.Spaceship(
        shader=shader,
        # diffuse=texture_dictionary['atlas_debug_diffuse'],
        # specular=texture_dictionary['atlas_debug_specular'],
        # emission=texture_dictionary['atlas_debug_emission'],
        diffuse=spaceship_parameters['diffuse'],
        specular=spaceship_parameters['specular'],
        emission=spaceship_parameters['emission'],

        # diffuse=texture_dictionary["penguin_diffuse"],
        # emission=texture_dictionary["penguin_emission"],
        # specular=texture_dictionary["penguin_specular"],
        dimensions=[5.0, 5.0],
        position=spaceship_parameters['position'],
        rotation_magnitude=[0.0, 0.0, -m.pi*0.5],
        number_of_sides=spaceship_parameters['number_of_sides'],
        # number_of_segments=spaceship_parameters['number_of_segments'],
        number_of_segments=spaceship_parameters['number_of_segments'],
        transform_x=spaceship_parameters['transform_x'],
        transform_z=spaceship_parameters['transform_z'],
        length_of_segment=spaceship_parameters['length_of_segment'],
        radius=3.0,
        scale=spaceship_parameters['scale']
    )
    ship_current = SpaceShip.Spaceship(model=spaceship, wallet=50)
    ship_current.set_velocity(velocity=[-1.5, 0.0, 0.0])
    ships.append(ship_current)

    ship_current.set_orders(
        orders=[['stuck']]
    )


my_fps = FPSCounter.FPSCounter(frame_interval=300.0, mute=True)


meshes = []
meshes += ships


def generate_new_ship():
    global spaceship
    spaceship = primatives.Spaceship(
        shader=shader,
        diffuse=spaceship_parameters['diffuse'],
        specular=spaceship_parameters['specular'],
        emission=spaceship_parameters['emission'],
        dimensions=[5.0, 5.0],
        position=ships[0].model.position,
        rotation_magnitude=ships[0].model.rotation_magnitude,
        rotation_axis=glm.vec3((0.0, 0.0, 1.0)),
        number_of_sides=spaceship_parameters['number_of_sides'],
        number_of_segments=spaceship_parameters['number_of_segments'],
        transform_x=spaceship_parameters['transform_x'],
        transform_z=spaceship_parameters['transform_z'],
        length_of_segment=spaceship_parameters['length_of_segment'],
        radius=3.0,
        scale=spaceship_parameters['scale'],

    )
    ships[0].model = spaceship


from GUI import Element, GUI, Character, TextBox

# test_gui = Element(position=(-.75, -0.75), scale=(0.25, 0.25), texture=texture_dictionary['whoa_diffuse'], atlas_size=2, atlas_coordinate=3)
text_boxes = []
test_gui = GUI(screen_size=(WIDTH, HEIGHT))

test_gui.add_button(
    shader=None,
    position=(-0.5, 0.5),
    position_mode='center',
    scale=(0.5, 0.5),
    texture=texture_dictionary['button_atlas'],
    atlas_size=2,
    atlas_coordinate=(2, 3),
    click_function=None,
    context_id='lorem',
    context_status=False,
    color=(1.0, 1.0, 1.0, 1.0)
)
text_button = test_gui.buttons[0]
text_button.add_text_box(
    texture=texture_dictionary['font_atlas'],
    font_size=0.35,
    color=(1.0, 1.0, 1.0, 1.0),
    centered=True,
    text='Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque in aliquam elit. Duis non eros augue.',
)

test_gui.add_button(
    shader=None,
    position=(0.75, 0.80),
    scale=(0.32, 0.05),
    texture=texture_dictionary['button_atlas'],
    atlas_size=2,
    atlas_coordinate=(2, 1),
    click_function=generate_new_ship,
    context_id='button',
    context_status=False,
    color=(1.0, 1.0, 1.0, 1.0)
)
text_button = test_gui.buttons[1]
text_button.add_text_box(
    texture=texture_dictionary['font_atlas'],
    font_size=0.35,
    color=(1.0, 1.0, 1.0, 1.0),
    text='Generate New Ship'
)

test_gui.add_button(
    shader=None,
    position=(0.80, 0.70),
    scale=(0.22, 0.05),
    texture=texture_dictionary['button_atlas'],
    atlas_size=2,
    atlas_coordinate=(2, 1),
    click_function=cycle_ship_texture,
    context_id='button',
    context_status=False,
    color=(1.0, 1.0, 1.0, 1.0)
)
text_button = test_gui.buttons[2]
text_button.add_text_box(
    texture=texture_dictionary['font_atlas'],
    font_size=0.35,
    color=(1.0, 1.0, 1.0, 1.0),
    text='Next Texture'
)
test_gui.add_button(
    shader=None,
    position=(0.85, 0.60),
    scale=(0.12, 0.05),
    texture=texture_dictionary['button_atlas'],
    atlas_size=2,
    atlas_coordinate=(2, 1),
    click_function=test_gui.switch_context_status,
    context_id='button',
    context_status=False,
    click_function_context_id='button',
    click_function_status_status=False,
    color=(1.0, 1.0, 1.0, 1.0)
)
text_button = test_gui.buttons[3]
text_button.add_text_box(
    texture=texture_dictionary['font_atlas'],
    font_size=0.35,
    color=(1.0, 1.0, 1.0, 1.0),
    text='Close'
)

test_gui.add_button(
    shader=None,
    position=(0.85, 0.90),
    scale=(0.12, 0.05),
    texture=texture_dictionary['button_atlas'],
    atlas_size=2,
    atlas_coordinate=(2, 1),
    click_function=test_gui.switch_context_status,
    context_id='context_on',
    context_status=True,
    click_function_context_id='button',
    click_function_status_status=True,
    color=(1.0, 1.0, 1.0, 1.0)
)
text_button = test_gui.buttons[4]
text_button.add_text_box(
    texture=texture_dictionary['font_atlas'],
    font_size=0.35,
    color=(1.0, 1.0, 1.0, 1.0),
    text='Options',
)

def change_text(text_box):
    """
    button-function for changing the text of a text box
    :param text_box:
    :return:
    """
    if text_box:
        text_box.update_text(text='wow, we changed the text', color=(1.0, 0.0, 0.0, 1.0))
    else:
        print(f'No textbox in change_text!')
# """Text Rendering"""
# test_char = TextBox(
#     shader=None,
#     texture=texture_dictionary['font_atlas'],
#     position=(-1.0, 1.0),
#     scale=(0.5, 0.5),
#     screen_size=(WIDTH, HEIGHT),
#     context_id='default',
#     text='Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.',
#     font_size=0.25,
#     width=1.5,
#     color=(0.8, 0.3, 0.25, 1.0),
# )
# test_gui.add_button(
#     shader=None,
#     position=(0.0, 0.0),
#     scale=(0.25, 0.25),
#     texture=texture_dictionary['whoa_diffuse'],
#     atlas_size=2,
#     atlas_coordinate=(2, 1),
#     click_function=change_text,
#     context_status=True,
#     click_function_text_box=text_button.text_box,
# )

test_gui.build_elements_list()







while not glfw.window_should_close(window):
    glfw.poll_events()
    do_movement(speed=100)
    my_fps.update()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    view = active_camera.get_view_matrix()
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)



    for mesh in meshes:
        mesh.draw(view=view)

    # for shape in shapes:
    #     shape.draw(view=view)

    """Mouse Hover on GUI"""
    if use_follow_cam:
        test_gui.button_update(position_mouse=glfw.get_cursor_pos(window), left_click=False, right_click=False)



    """draw new ship"""
    if make_new_ship:
        make_new_ship = False
        generate_new_ship()

    # draw the point light cube
    # for light in debug_plcs:
    #     light.draw(view)
    glUseProgram(shader)

    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

    projection = pyrr.matrix44.create_perspective_projection_matrix(45, WIDTH / HEIGHT, 0.1, 2050)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

    # pass cam position for specular light
    glUniform3fv(view_pos_loc, 1, list(active_camera.camera_pos))
    point_light_position_loc = glGetUniformLocation(shader, "point_light.position")
    glUniform1f(glGetUniformLocation(shader, "material.shininess"), 4.0)
    glUniform3fv(glGetUniformLocation(shader, "point_lights[0].position"), 1, debug_plcs[0].get_pos())
    glUniform3fv(glGetUniformLocation(shader, "point_lights[0].diffuse"), 1, debug_plcs[0].get_diffuse())
    glUniform3fv(glGetUniformLocation(shader, "point_lights[0].ambient"), 1, debug_plcs[0].get_ambient())
    glUniform3fv(glGetUniformLocation(shader, "point_lights[0].specular"), 1, debug_plcs[0].get_specular())
    glUniform1f(glGetUniformLocation(shader, "point_lights[0].constant"), debug_plcs[0].get_constant())
    glUniform1f(glGetUniformLocation(shader, "point_lights[0].linear"), debug_plcs[0].get_linear())
    glUniform1f(glGetUniformLocation(shader, "point_lights[0].quadratic"), debug_plcs[0].get_quadratic())

    #second light for mesh viewing
    glUniform3fv(glGetUniformLocation(shader, "point_lights[1].position"), 1, debug_plcs[1].get_pos())
    glUniform3fv(glGetUniformLocation(shader, "point_lights[1].diffuse"), 1, debug_plcs[1].get_diffuse())
    glUniform3fv(glGetUniformLocation(shader, "point_lights[1].ambient"), 1, debug_plcs[1].get_ambient())
    glUniform3fv(glGetUniformLocation(shader, "point_lights[1].specular"), 1, debug_plcs[1].get_specular())
    glUniform1f(glGetUniformLocation(shader, "point_lights[1].constant"), debug_plcs[1].get_constant())
    glUniform1f(glGetUniformLocation(shader, "point_lights[1].linear"), debug_plcs[1].get_linear())
    glUniform1f(glGetUniformLocation(shader, "point_lights[1].quadratic"), debug_plcs[1].get_quadratic())

    # spotlight
    glUniform3fv(glGetUniformLocation(shader, "spot_light.position"), 1, list(active_camera.camera_pos))
    glUniform3fv(glGetUniformLocation(shader, "spot_light.direction"), 1, list(active_camera.camera_front))
    glUniform3fv(glGetUniformLocation(shader, "spot_light.diffuse"), 1, [0.0]*3)
    glUniform3fv(glGetUniformLocation(shader, "spot_light.ambient"), 1, [0.0]*3)
    glUniform3fv(glGetUniformLocation(shader, "spot_light.specular"), 1, [0.0]*3)
    glUniform1f(glGetUniformLocation(shader, "spot_light.cut_off"), glm.cos(glm.radians(12.5)))
    glUniform1f(glGetUniformLocation(shader, "spot_light.outer_cut_off"), glm.cos(glm.radians(45.0)))
    glUniform1f(glGetUniformLocation(shader, "spot_light.constant"), 1.0)
    glUniform1f(glGetUniformLocation(shader, "spot_light.linear"), 0.00003)
    glUniform1f(glGetUniformLocation(shader, "spot_light.quadratic"), 0.00007)

    if write_to_gif:
        window_as_numpy_arr = np.uint8(glReadPixels(0, 0, 800, 800, GL_RGB, GL_FLOAT)*255.0)
        window_as_numpy_arr = np.flip(window_as_numpy_arr, 0)
        window_as_PIL_image = Image.fromarray(window_as_numpy_arr)
        images.append(window_as_PIL_image)



    """GUI TESTING"""
    test_gui.draw()
    for text_box in text_boxes:
        text_box.draw()
    """Text Rendering"""
    # test_char.draw()


    glUseProgram(shader)


    glfw.swap_buffers(window)

if write_to_gif:
    images[0].save(
            'workshop.gif',
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=20,
            loop=0
        )
glfw.terminate()
