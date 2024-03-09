"""
4th February, 2024
This files contains a code for creating and displaying our proc-gen spaceships

-[V]upload to github(initial)
-[V]Emission Map in our default shader
-[]clear out old WFC code
-[]basic spaceship generator (in primative_meshes)
-[]some sort of propulsion trail on our ship
-[]a starry skybox around the ship
-[]share and post on reddit, discord
-[]a video documenting our process

"""
import math

import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
from TextureLoader import load_texture
from Camera import Camera, FollowCamera
import math as m
import random as r
import numpy as np
import glm
from PIL import Image
import PointLightCube as plc
import ProceduralMesh as primatives
import FPSCounter

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
# the keyboard input callback
def key_input_clb(window, key, scancode, action, mode):
    global left, right, forward, backward, make_new_surface, player_left, player_right, player_forward, \
        player_backward, cycle_camera_mode, yaw_counterclockwise, yaw_clockwise, write_to_gif
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
    // phase 3: spot light
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

projection = pyrr.matrix44.create_perspective_projection_matrix(45, WIDTH / HEIGHT, 0.1, 100)

model_loc = glGetUniformLocation(shader, "model")
proj_loc = glGetUniformLocation(shader, "projection")
view_loc = glGetUniformLocation(shader, "view")

view_pos_loc = glGetUniformLocation(shader, "view_pos")  # cam pos for specular lighting

glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

# create the light cube
my_plc = plc.PointLightCube(pos=[0.0, 0.0, 0.0], ambient=[0.0]*3, diffuse=[0.0]*3,
                            specular=[0.0]*3)
debug_plcs = []
# debug_plcs.append(plc.PointLightCube(
#     pos=[0.0, 5.0, 5.0],
#     ambient=[0.25]*3,
#     diffuse=[0.7]*3,
#     specular=[0.6]*3
# ))
#nighty night mode
debug_plcs.append(plc.PointLightCube(
    pos=[0.0, 5.0, 5.0],
    ambient=[0.0]*3,
    diffuse=[0.0]*3,
    specular=[0.0]*3
))

#seed random
r.seed(1903)

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
load_texture("Textures/debug_texture_atlas.png", tile_textures[0])
load_texture("Textures/debug_texture_atlas.png", tile_textures[1])
load_texture("Textures/debug_texture_atlas.png", tile_textures[2])
load_texture("Textures/debug_texture_atlas.png", tile_textures[3])
load_texture("Textures/debug_texture_atlas.png", tile_textures[4])
load_texture("Textures/debug_texture_atlas.png", tile_textures[5])
load_texture("Textures/debug_texture_atlas.png", tile_textures[6])
load_texture("Textures/spaceship_texture_atlas_1.png", tile_textures[7])
load_texture("Textures/spaceship_texture_atlas_1_specular.png", tile_textures[8])
load_texture("Textures/spaceship_texture_atlas_1_emission.png", tile_textures[9])
load_texture("Textures/debug_texture_atlas_2.png", tile_textures[10])
load_texture("Textures/debug_texture_atlas_2_emission.png", tile_textures[11])
load_texture("Textures/debug_texture_atlas_2_specular.png", tile_textures[12])


texture_dictionary = {
    "pink": tile_textures[3],
    "red": tile_textures[2],
    "stone": tile_textures[1],
    "atlas_debug": tile_textures[4],
    "spaceship_diffuse": tile_textures[7],
    "spaceship_specular": tile_textures[8],
    "spaceship_emission": tile_textures[9],
    "atlas_debug_diffuse": tile_textures[10],
    "atlas_debug_emission": tile_textures[11],
    "atlas_debug_specular": tile_textures[12],
}

spaceship = primatives.Spaceship(
    shader=shader,
    diffuse=texture_dictionary['atlas_debug_diffuse'],
    specular=texture_dictionary['atlas_debug_specular'],
    emission=texture_dictionary['atlas_debug_emission'],
    # diffuse=texture_dictionary['spaceship_diffuse'],
    # specular=texture_dictionary['spaceship_specular'],
    # emission=texture_dictionary['spaceship_emission'],
    dimensions=[5.0, 5.0],
    position=[25.0, 10.0, 15.0],
    rotation_magnitude=-m.pi*0.5,
    rotation_axis=glm.vec3([0.0, 0.0, 1.0]),
)

player_model = primatives.SegmentedPrismBevelPolygonCornerTest(
    shader=shader,
    diffuse=tile_textures[3],
    specular=tile_textures[3],
    shininess=32.0,
    position=[0.0, 0.0, 10.0],
    dimensions=[5.0, 10.0, 1.0],
    rotation_axis=[0.0, 1.0, 0.0],
    scale=[0.5, 0.5, 0.5],
    sides=4,
    segments=8,
    bevel_depths=[0.0, -0.2, 0.9, 0.9, -0.9],
    border_sizes=[0.2, 0.2, 0.2, 0.2, 0.1],
    depth=0
)

bezier_cube = primatives.NPrismBezierCut(
    shader=shader,
    diffuse=texture_dictionary["atlas_debug_diffuse"],
    specular=texture_dictionary["atlas_debug"],
    shininess=32.0,
    dimensions=[2.0],
    position=[0.0, 5.0, -5.0],
    rotation_axis=glm.vec3([0.0, 1.0, 0.0]),
    rotation_magnitude=m.pi,
    scale=glm.vec3([5.0]*3),
    sides=5
)

testing_model = primatives.Prism(
    shader=shader,
    diffuse=texture_dictionary['pink'],
    specular=texture_dictionary['pink'],
    dimensions=[2.0, 2.0],
    position=[0.0]*3,
    sides = 4,
)
testing_model_polygon = primatives.Polygon(
    shader=shader,
    diffuse=texture_dictionary['stone'],
    specular=texture_dictionary['stone'],
    dimensions=[20.0, 2.0],
    position=[5.0, 0.0, 0.0],
    sides = 12,
)

testing_primative_mesh_emission = primatives.PrimativeMeshEmission(
    shader=shader,
    diffuse=texture_dictionary['atlas_debug_diffuse'],
    specular=texture_dictionary['atlas_debug_specular'],
    emission=texture_dictionary['atlas_debug_emission'],
    dimensions=[10.0, 10.0],
    position=[20.0, 10.0, 0.0],
)



my_fps = FPSCounter.FPSCounter(frame_interval=300.0)

# the main application loop
while not glfw.window_should_close(window):
    glfw.poll_events()
    do_movement(speed=10)
    my_fps.update()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    ct = glfw.get_time()

    view = cam.get_view_matrix()
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

    """TESTING Proc Gen"""
    # bezier_cube.draw(view=view)
    # testing_model.draw(view=view)
    # testing_model_polygon.draw(view=view)
    # player_model.rotate_over_time()
    # player_model.draw(view=view)
    # testing_primative_mesh_emission.draw(view=view)
    # spaceship.rotate_over_time(speed=0.3)
    spaceship.draw(view=view)

    # draw the point light cube
    # my_plc.draw(view)
    for light in debug_plcs:
        light.draw(view)
    glUseProgram(shader)

    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

    projection = pyrr.matrix44.create_perspective_projection_matrix(45, WIDTH / HEIGHT, 0.1, 150)
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
    glUniform3fv(glGetUniformLocation(shader, "spot_light.diffuse"), 1, [1.0]*3)
    glUniform3fv(glGetUniformLocation(shader, "spot_light.ambient"), 1, [0.0]*3)
    glUniform3fv(glGetUniformLocation(shader, "spot_light.specular"), 1, [1.0]*3)
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