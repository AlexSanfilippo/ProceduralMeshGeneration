"""
started: April 19th, 2024
update: April 19th, 2024

goals:
    -[V]render quad on screen
    -[V]textured quad
    -[V]control scale and position
    -[V]transparent textures
    -[V]texture atlasing
    -[V]Multiple GUI Elements
        -static elements.icons
            -displays, ie, not buttons
        -want GUI class that holds several GUI elements
            instantiate GUI class
            add element to class
            GUI.draw
                -draws all elements
                    -separate draw calls for now
                        -so just a loop

    -[V]buttons
        -subclass existing gui element class
        -[V]Check if mouse is inside button
        -[V]Check if mouse is clicked
        -[V]button to spawn ships
            -ie, tie button into application
    -[V]Change texture on mouse hover (hover events)
        -send mouse position into draw
            -allow each element to figure itself out
    -[]activate/deactivate button by clicking other buttons
"""
import math
import glm
from OpenGL.GL import *
import numpy as np
from OpenGL.GL.shaders import compileProgram, compileShader

from TextureLoader import load_texture

vertex_src = """
# version 330

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texture;

out vec2 texture_coordinates;

void main()
{
    texture_coordinates = texture;
    gl_Position = vec4(position, 0.f, 1.f);
}
"""

fragment_src = """
#version 330 core

in vec2 texture_coordinates;
out vec4 fragment_color;
uniform sampler2D gui_texture;

void main()
{    
    
    //fragment_color = vec4(0.0, 0.7, 0.7, 0.5f);
    fragment_color = texture(gui_texture, texture_coordinates);
}
"""

vertex_src_reference = """
# version 330

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;
layout(location = 2) in vec2 a_texture;

uniform mat4 rotation;

out vec3 v_color;
out vec2 v_texture;
void main()
{
    gl_Position = rotation * vec4(a_position, 1.0);
    v_color = a_color;
    v_texture = a_texture;
}
"""

fragment_src_reference = """
# version 330

in vec3 v_color;
in vec2 v_texture;

out vec4 out_color;

uniform sampler2D s_texture;

void main()
{
    out_color =  texture(s_texture, v_texture);  // vec4(v_color, 1.0);
}
"""

#create default shaders and textures
# shader_default = compileProgram(
#     compileShader(
#         vertex_src,
#         GL_VERTEX_SHADER
#     ),
#     compileShader(
#         fragment_src,
#         GL_FRAGMENT_SHADER
#     ),
# )


class GUI:
    """
    Create GUI For application interaction with mouse, or displaying information
    """
    def __init__(
            self,
            screen_size=(800, 400)
    ):
        self.screen_size = screen_size
        self.elements = []
        self.buttons = []

    def add_element(
            self,
            shader=None,
            texture=None,
            position=(0.0, 0.0),
            scale=(0.5, 0.5),
            atlas_size=1,
            atlas_coordinate=0
    ):
        self.elements.append(
            Element(
                shader=shader,
                texture=texture,
                position=position,
                scale=scale,
                screen_size=self.screen_size,
                atlas_size=atlas_size,
                atlas_coordinate=atlas_coordinate
            )
        )

    def draw(self):
        for element in self.elements + self.buttons:
            element.draw()

    def add_button(
        self,
        shader=None,
        texture=None,
        position=(0.0, 0.0),
        scale=(0.5, 0.5),
        atlas_size=1,
        atlas_coordinate=(0,),
        click_function=None,
    ):
        self.buttons.append(
            Button(
                shader=shader,
                texture=texture,
                position=position,
                scale=scale,
                screen_size=self.screen_size,
                atlas_size=atlas_size,
                atlas_coordinate=atlas_coordinate,
                click_function=click_function
            )
        )

    def button_update(self, position_mouse, left_click, right_click):
        """
        returns list of 0 or 1 for button being clicked
        :param position_mouse:
        :param left_click:
        :param right_click:
        :return:
        """
        position_mouse_normalized = glm.vec2(
            (
                position_mouse[0]/self.screen_size[0],
                position_mouse[1]/self.screen_size[1]
            )
        )

        for button in self.buttons:
            button.update(position_mouse=position_mouse_normalized, left_click=left_click, right_click=right_click)


class Element:

    def __init__(
            self,
            shader=None,
            texture=None,
            position=(0.0, 0.0),
            scale=(0.5, 0.5),
            screen_size=(800, 400),
            atlas_size=1,
            atlas_coordinate=0,
    ):
        if shader == None:
            shader_default = compileProgram(
                compileShader(vertex_src, GL_VERTEX_SHADER),
                compileShader(fragment_src, GL_FRAGMENT_SHADER)
            )
            self.shader = shader_default

        else:
            self.shader = shader
        if texture == None:
            gui_textures = glGenTextures(2)
            load_texture("Textures/debug_diffuse_coordinates.png", gui_textures[0])
            self.texture = gui_textures[0]
        else:
            self.texture = texture
        self.position = glm.vec2(position)
        self.scale = scale
        self.screen_size = screen_size
        self.screen_size = screen_size
        self.atlas_size = atlas_size
        self.atlas_coordinate = atlas_coordinate
        self.vertices = self.generate_vertices()
        self.buffer_setup()

    def generate_vertices(self):

        text_coords = self.get_texture_coordinates_atlas()

        #OpenGL Screen Coordinates go from -1,-1 (lower left) to 1,1 (upper right)
        return np.array(
            [
                # upper left
                -1.0 * self.scale[0] + self.position[0], 1.0 * self.scale[1] + self.position[1], text_coords[0].x, text_coords[0].y,
                # lower left
                -1.0 * self.scale[0] + self.position[0], -1.0 * self.scale[1] + self.position[1], text_coords[1].x, text_coords[1].y,
                #upper right
                1.0 * self.scale[0] + self.position[0], 1.0 * self.scale[1] + self.position[1], text_coords[2].x,   text_coords[2].y,
                #lower right
                1.0 * self.scale[0] + self.position[0], -1.0 * self.scale[1] + self.position[1], text_coords[3].x,  text_coords[3].y,
            ],
            dtype=np.float32
        )

    def buffer_setup(self):
        # quad VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        # quad position vertices (vertex attribute)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # quad texture coords
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 4, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)

        self.model_loc = glGetUniformLocation(self.shader, "model")

    def draw(self):
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)

        # rotate, translate, and scale
        # model = glm.mat4(1.0)
        # model = glm.translate(model, self.position)
        # todo: this rotation smells fishy
        # model = glm.rotate(model, self.rotation_magnitude.x, self.rotation_axis)
        # model = glm.scale(model, self.scale)
        # glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, glm.value_ptr(model))

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

    def get_texture_coordinates_atlas(self):
        texture_coordinates = [
            glm.vec2(0.0, 1.0),
            glm.vec2(0.0, 0.0),
            glm.vec2(1.0, 1.0),
            glm.vec2(1.0, 0.0)
        ]

        # texture atlas index into row and column indices
        column_index = float(self.atlas_coordinate % self.atlas_size)
        row_index = 1.0 - math.floor(self.atlas_coordinate / self.atlas_size)

        # row and column indices into lower and upper bounds of texture coords (4 total)
        # column (x-axis)lower and upper
        lower_texture_coord_x_axis = 1.0 / self.atlas_size * column_index
        upper_texture_coord_x_axis = 1.0 / self.atlas_size * (column_index + 1.0)
        # assuming square textures
        subtexture_magnitude = upper_texture_coord_x_axis - lower_texture_coord_x_axis
        lower_texture_coord_y_axis = 1.0 / self.atlas_size * row_index
        upper_texture_coord_y_axis = 1.0 / self.atlas_size * (row_index + 1.0)
        # update the current vertices' text coords
        for text_coord in texture_coordinates:
            text_coord[0] = \
                lower_texture_coord_x_axis \
                + text_coord[0] \
                * subtexture_magnitude
            text_coord[1] = \
                lower_texture_coord_y_axis \
                + text_coord[1] \
                * subtexture_magnitude
        return texture_coordinates

class Button(Element):
    """
    Clickable GUI Elements
    """

    def __init__(
        self,
        shader=None,
        texture=None,
        position=(0.0, 0.0),
        scale=(0.5, 0.5),
        screen_size=(800, 400),
        atlas_size=1,
        atlas_coordinate=(0, 0),
        click_function=None
    ):
        super().__init__(
            shader=shader,
            texture=texture,
            position=position,
            scale=scale,
            screen_size=screen_size,
            atlas_size=atlas_size,
            atlas_coordinate=atlas_coordinate[0],
        )
        self.generate_bounds()
        self.click_function = click_function
        self.atlas_coordinate_on = atlas_coordinate[0]
        self.atlas_coordinate_off = atlas_coordinate[1]
    def generate_bounds(self):
        """
        Define vertical and horizontal limits that allow for testing if cursor is
        inside or outside of the button
        """
        self.bounds = {
            'vertical_lower': 1.0 - ((self.scale[1] + self.position[1]) * 0.5 + 0.5),
            'vertical_upper': 1.0 - ((-1.0 * self.scale[1] + self.position[1]) * 0.5 + 0.5),
            'horizontal_upper': (self.scale[0] + self.position[0]) * 0.5 + 0.5,
            'horizontal_lower': (-1.0 * self.scale[0] + self.position[0]) * 0.5 + 0.5,
        }

    def check_mouse_hover(self, position_mouse):
        """
        check if the mouse is inside this element
        :return: Bool
        """
        if position_mouse.x < self.bounds['horizontal_upper'] \
            and position_mouse.x > self.bounds['horizontal_lower'] \
            and position_mouse.y > self.bounds['vertical_lower'] \
            and position_mouse.y < self.bounds['vertical_upper']:
            print(f'mouse in button! bounds={self.bounds}')
            return True
        else:
            return False
    def check_mouse_click(self, left_click):
        """
        Check if the mouse is clicked while inside this element
        :return:
        """
        return left_click

    def update(self, position_mouse, left_click=False, right_click=False):
        """
        check for mouse input and react
        :return:
        """
        if self.check_mouse_hover(position_mouse=position_mouse):
            self.atlas_coordinate = self.atlas_coordinate_off
            self.vertices = self.generate_vertices()
            self.buffer_setup()
            if self.check_mouse_click(left_click):
                print('Mouse Clicked this Button!')
                if self.click_function:
                    self.click_function()
                else:
                    print("No click function!")
        else:
            self.atlas_coordinate = self.atlas_coordinate_on
            self.vertices = self.generate_vertices()
            self.buffer_setup()

