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
    -[V]activate/deactivate button by clicking other buttons
"""
import gc
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
uniform vec4 color;

void main()
{    
    
    //fragment_color = vec4(0.7, 0.1, 0.5, 0.5f);
    fragment_color = color * texture(gui_texture, texture_coordinates);
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
        self.context_id_to_status = {}
        self.context_id_to_element = {}

    def add_element(
            self,
            shader=None,
            texture=None,
            position=(0.0, 0.0),
            scale=(0.5, 0.5),
            atlas_size=1,
            atlas_coordinate=0,
            context_status=True,
            context_id='default'
    ):
        element = Element(
            shader=shader,
            texture=texture,
            position=position,
            scale=scale,
            screen_size=self.screen_size,
            atlas_size=atlas_size,
            atlas_coordinate=atlas_coordinate,
            context_id=context_id,
        )
        self.elements.append(element)
        self.update_context_maps(element=element, status=context_status)

    def add_text_element(
            self,
            font_texture,
            shader=None,
            texture=None,
            position=(0.0, 0.0),
            scale=(0.5, 0.5),
            atlas_size=1,
            atlas_coordinate=0,
            context_status=True,
            context_id='default',
            text='',
            font_color=(1.0, 1.0, 1.0, 1.0),
            font_size=1.0,
    ):
        element = Element(
            shader=shader,
            texture=texture,
            position=position,
            scale=scale,
            screen_size=self.screen_size,
            atlas_size=atlas_size,
            atlas_coordinate=atlas_coordinate,
            context_id=context_id,
        )
        element.add_text_box(
            texture=font_texture,
            text=text,
            color=font_color,
            font_size=font_size,

        )
        self.elements.append(element)
        self.update_context_maps(element=element, status=context_status)


    def draw(self):
        #TODO: draw all elements at once! (batch rendering)
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
        context_id='default',
        context_status=True,
        color=(1.0, 1.0, 1.0, 1.0),
        position_mode='center',
        **click_function_kwargs,
    ):
        button = Button(
            shader=shader,
            texture=texture,
            position=position,
            scale=scale,
            screen_size=self.screen_size,
            atlas_size=atlas_size,
            atlas_coordinate=atlas_coordinate,
            click_function=click_function,
            context_id=context_id,
            color=glm.vec4(color),
            position_mode=position_mode,
            click_function_kwargs=click_function_kwargs,
        )
        self.buttons.append(button)
        self.update_context_maps(element=button, status=context_status)

    def add_text_button(
        self,
        font_texture,
        shader=None,
        texture=None,
        position=(0.0, 0.0),
        scale=(0.5, 0.5),
        atlas_size=1,
        atlas_coordinate=(0,),
        click_function=None,
        context_id='default',
        context_status=True,
        color=(1.0, 1.0, 1.0, 1.0),
        position_mode='center',
        text='',
        font_size=0.35,
        font_color=(1.0, 1.0, 1.0, 1.0),
        **click_function_kwargs,
    ):
        button = Button(
            shader=shader,
            texture=texture,
            position=position,
            scale=scale,
            screen_size=self.screen_size,
            atlas_size=atlas_size,
            atlas_coordinate=atlas_coordinate,
            click_function=click_function,
            context_id=context_id,
            color=glm.vec4(color),
            position_mode=position_mode,
            click_function_kwargs=click_function_kwargs,
        )
        button.add_text_box(
            texture=font_texture,
            font_size=font_size,
            color=font_color,
            text=text
        )
        self.buttons.append(button)
        self.update_context_maps(element=button, status=context_status)


    def button_update(self, position_mouse, left_click, right_click):
        """
        tell buttons to check for input (mouse hover or click)
        :param position_mouse:
        :param left_click:
        :param right_click:
        :return: No Return
        """
        position_mouse_normalized = glm.vec2(
            (
                position_mouse[0]/self.screen_size[0],
                position_mouse[1]/self.screen_size[1]
            )
        )

        for button in self.buttons:
            button.update(position_mouse=position_mouse_normalized, left_click=left_click, right_click=right_click)

    def update_context_maps(self, element, status=True):
        if element.context_id in self.context_id_to_status.keys():
            self.context_id_to_status[element.context_id] = status
            self.context_id_to_element[element.context_id].append(element)
        else:
            self.context_id_to_status[element.context_id] = status
            self.context_id_to_element[element.context_id] = [element]

    def build_elements_list(self):
        self.elements = []
        self.buttons = []
        for context_id, status in self.context_id_to_status.items():
            if status:
                elements = self.context_id_to_element[context_id]
                for element in elements:
                    if type(element) == Element:
                        self.elements.append(element)
                    else:
                        self.buttons.append(element)

    def switch_context_status(self, context_id, status):
        """
        turn a context on or off
        :param context_id: id of the element to turn on/off
        :param status: which status to switch it to
        """
        self.context_id_to_status[context_id] = status
        self.build_elements_list()

    def toggle_context_status(self, context_id):
        self.switch_context_status(
            context_id=context_id,
            status=not(self.context_id_to_status[context_id]),
        )


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
            context_id='default',
            color=glm.vec4(1.0, 1.0, 1.0, 1.0),
    ):
        self.color_loc = None
        self.model_loc = None
        self.VBO = None
        self.VAO = None
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
        self.context_id = context_id
        self.color = color
        self.vertices_count = 4
        self.vertices = self.generate_vertices()
        self.buffer_setup()
        self.text_box = None

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

    def clean_up(self):
        glDeleteVertexArrays(1, [self.VAO])
        glDeleteBuffers(1, [self.VBO])
        del self.model_loc
        del self.color_loc
        gc.collect()

    def buffer_setup(self):
        # quad VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        # quad position vertices (vertex attribute)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, self.vertices.itemsize * self.vertices_count, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # quad texture coords
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, self.vertices.itemsize * self.vertices_count, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)

        self.model_loc = glGetUniformLocation(self.shader, "model")
        self.color_loc = glGetUniformLocation(self.shader, "color")

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
        glUniform4fv(self.color_loc, 1, glm.value_ptr(self.color))

        glEnable(GL_BLEND)
        glDisable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, self.vertices_count)
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        if self.text_box:
            self.text_box.draw()

    def update_text(self, text=None, color=None):
        self.text_box.update_text(text=text, color=color)

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

    def add_text_box(self, texture, text="DEFAULT TEXT", color=(1.0, 1.0, 1.0, 1.0), font_size=1.0, width=1.0, centered=True):
        self.text_box = TextBox(
            texture=texture,
            position=(self.position.x, self.position.y),
            scale=self.scale,
            screen_size=self.screen_size,
            context_id=self.context_id,
            text=text,
            font_size=font_size,
            width=self.scale[0]*2*width,
            color=glm.vec4(color),
            centered=centered,
            )


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
        click_function=None,
        context_id='default',
        color=(1.0, 1.0, 1.0, 1.0),
        position_mode='center',
        click_function_kwargs=None,
    ):
        """
        :param shader:
        :param texture:
        :param position: tuple of length 3.
        :param scale: between 0 and 1, tuple of length 2
        :param screen_size: OpenGL Window dimensions
        :param atlas_size: length of atlas (square atlas only)
        :param atlas_coordinate: texture atlas coordinate.  tuple of 2 items
        :param click_function: function called upon click
        :param context_id: context for turning elements on/off
        :param color: pass vec4 color to shader
        :param position_mode: position by default is center. options: bottom_left, bottom_right, top_left, top_right
        :param click_function_kwargs: args to pass to function.
        """
        position = list(position)
        if position_mode == 'center':
            pass
        elif position_mode == 'top_left':
            position[0] += scale[0]
            position[1] -= scale[1]
        elif position_mode == 'top_right':
            position[0] -= scale[0]
            position[1] -= scale[1]
        elif position_mode == 'bottom_left':
            position[0] += scale[0]
            position[1] += scale[1]
        elif position_mode == 'bottom_right':
            position[0] -= scale[0]
            position[1] += scale[1]

        super().__init__(
            shader=shader,
            texture=texture,
            position=position,
            scale=scale,
            screen_size=screen_size,
            atlas_size=atlas_size,
            atlas_coordinate=atlas_coordinate[0],
            context_id=context_id,
            color=color
        )
        self.generate_bounds()
        self.click_function = click_function
        self.atlas_coordinate_on = atlas_coordinate[0]
        self.atlas_coordinate_off = atlas_coordinate[1]
        self.click_function_kwargs = click_function_kwargs

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
            self.clean_up()
            self.buffer_setup()
            if self.check_mouse_click(left_click):
                if self.click_function:
                    if self.click_function_kwargs:
                        self.click_function(*list(self.click_function_kwargs.values()))
                    else:
                        self.click_function()
                else:
                    print("WARNING: No click function!")
        else:
            self.atlas_coordinate = self.atlas_coordinate_on
            self.vertices = self.generate_vertices()
            self.clean_up()
            self.buffer_setup()


class Character(Element):
    """
    Render a single quad with a given character
    """

    def __init__(
        self,
        shader=None,
        texture=None,
        position=(0.0, 0.0),
        scale=(0.5, 0.5),
        screen_size=(800, 400),
        context_id='default',
        text=None,
        font_size=1.0,
        width=1.0,
        color=(1.0, 1.0, 1.0, 1.0),
        centered=False,
    ):
        """

        :param shader:
        :param texture:
        :param position:
        :param scale: Not used here, but leftover from Element
        :param screen_size: Pixel Width, Height of screen
        :param context_id: For opening/closing GUI contexts
        :param text: The text to display
        :param font_size: Functions as multiplier.
        :param width: Width as fraction of screen [0.0, 1.0]
        """

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
        self.position = glm.vec2((position[0] - scale[0], position[1] + scale[1]))
        self.scale = scale
        self.screen_size = screen_size
        self.screen_size = screen_size
        self.context_id = context_id
        self.text = text
        self.font_size = font_size
        self.vertices_count = 4
        self.color = glm.vec4(color)
        self.centered = centered
        self.characters = self.get_characters()
        self.width = width
        self.box_right = self.position.x + width
        self.vertices = self.generate_vertices()
        self.buffer_setup()

    def get_characters(self):
        file_font = open('C://Users//LENOVO//PycharmProjects//ProceduralMeshGeneration//Fonts//my_font.fnt', 'r')
        characters = dict()
        line_count = 0
        for line in file_font:
            line = line[:-1]
            if line_count < 4:
                line_count += 1
                continue
            else:
                line_as_list = line.split()
                line_type = line_as_list[0]
                line_values = line_as_list[1:]
                if line_type == 'kerning':
                    break
                character_values = dict()
                for pair in line_values[1:]:
                    pair = pair.split(sep='=')
                    character_values[pair[0]] = int(pair[1])
                char_id = line_values[0].split(sep='=')[1]
                characters[chr(int(char_id))] = character_values
        return characters

    def generate_vertices(self):

        char_info = self.characters[self.text]
        width = char_info['width']/512
        height = char_info['height']/512
        text_coords_upper_left = glm.vec2(char_info['x'], 1.0 - char_info['y'])/512
        text_coords_upper_right = glm.vec2(text_coords_upper_left.x + width, text_coords_upper_left.y)
        text_coords_lower_right = glm.vec2(text_coords_upper_left.x + width, text_coords_upper_left.y - height)
        text_coords_lower_left = glm.vec2(text_coords_upper_left.x, text_coords_upper_left.y - height)
        text_coords = [text_coords_upper_left, text_coords_lower_left, text_coords_upper_right, text_coords_lower_right]

        #OpenGL Screen Coordinates go from -1,-1 (lower left) to 1,1 (upper right)
        return np.array(
            [
                # upper left
                -1.0 * width * self.font_size + self.position[0], 1.0 * height * self.font_size + self.position[1], text_coords[0].x, text_coords[0].y,
                # lower left
                -1.0 * width * self.font_size + self.position[0], -1.0 * height * self.font_size + self.position[1], text_coords[1].x, text_coords[1].y,
                #upper right
                1.0 * width * self.font_size + self.position[0], 1.0 * height * self.font_size + self.position[1], text_coords[2].x,   text_coords[2].y,
                #lower right
                1.0 * width * self.font_size + self.position[0], -1.0 * height * self.font_size + self.position[1], text_coords[3].x,  text_coords[3].y,
            ],
            dtype=np.float32
        )


class TextBox(Character):
    """
    Much like Character, except takes a whole string and produces multiple quads
    -Uses OpenGL screen Coordinates, ie (-1,-1) in South West, (1,1) in North East
        -so a width of 2.0 would have text box cover whole screen

    """

    def generate_vertices(self):
        vertices = []
        vertices_line = []
        self.box_left = self.position[0]
        box_center_horizontal = (self.box_right + self.box_left) / 2.0
        position_initial = self.position.__copy__()
        text = self.text

        # for index, char in enumerate(text):
        index_char = 0
        loop_count = 0
        history = {}
        history_word = {}
        while index_char < len(text):
            # loop_count += 1
            # if loop_count > 100:
            #     break
            if index_char not in set(history.keys()):
                history[index_char] = 0
            loop_count += 1
            char = text[index_char]
            char_info = self.characters[char]
            width = (char_info['width'] / 512)
            height = (char_info['height'] / 512)
            x_offset = (char_info['xoffset'] / 512)
            y_offset = (char_info['yoffset'] / 512)
            x_advance = (char_info['xadvance'] / 512)
            text_coords_upper_left = glm.vec2(char_info['x'], 1.0 - char_info['y']) / 512
            text_coords_upper_right = glm.vec2(text_coords_upper_left.x + width, text_coords_upper_left.y)
            text_coords_lower_right = glm.vec2(text_coords_upper_left.x + width, text_coords_upper_left.y - height)
            text_coords_lower_left = glm.vec2(text_coords_upper_left.x, text_coords_upper_left.y - height)
            text_coords = [text_coords_upper_left, text_coords_lower_left, text_coords_upper_right, text_coords_lower_right]

            width = (char_info['width'] / 512) * self.font_size
            height = (char_info['height'] / 512) * self.font_size
            x_offset = (char_info['xoffset'] / 512) * self.font_size
            y_offset = (char_info['yoffset'] / 512) * self.font_size
            x_advance = (char_info['xadvance'] / 512) * self.font_size

            upper_left = [(self.position[0] + x_offset),  (self.position[1] - y_offset), text_coords[0].x, text_coords[0].y, ]
            lower_left = [(self.position[0] + x_offset),  (self.position[1] - y_offset - height), text_coords[1].x, text_coords[1].y, ]
            upper_right = [(self.position[0] + width + x_offset),  (self.position[1] - y_offset), text_coords[2].x, text_coords[2].y, ]
            lower_right = [(self.position[0] + width + x_offset),  (self.position[1] - y_offset - height), text_coords[3].x, text_coords[3].y, ]

            positions = upper_left + lower_left + upper_right + lower_right + upper_right + lower_left
            vertices_line += positions

            self.position[0] += x_advance

            #Check if need newline and handle
            if self.position[0] > self.box_right:
                if char == ' ':
                    self.position[0] = self.box_left
                    self.position[1] -= 0.25 * self.font_size
                else:
                    #if haven't tried newlining at this index yet
                    index_back_check = index_char
                    char_back_check = char
                    while char_back_check != ' ':
                        char_back_check = text[index_back_check]
                        #remove char's quad
                        vertices_line = vertices_line[:-4*6]
                        #move onto previous char
                        index_back_check -= 1
                    #go back to beginning of word
                    index_char = index_back_check + 1
                    index_first_letter = index_char + 1
                    if index_first_letter in set(history_word.keys()):
                        history_word[index_first_letter] += 1
                    else:
                        history_word[index_first_letter] = 1
                    #move cursor down to next line, and to far left side
                    if history_word[index_first_letter] <= 1:
                        self.position[0] = self.box_left
                        self.position[1] -= 0.25 * self.font_size
                    else:
                        #if we've already tried newlining this word, just have to split the word
                        raise Exception(f'A word at index {index_first_letter} in the text is longer than the text box.')

                        # index_char = index_char_original - 1
                        # self.position[0] = self.box_left
                        # self.position[1] -= 0.25 * self.font_size
                if self.centered:
                    self.center_text_horizontal(box_center_horizontal, vertices_line)

                vertices += vertices_line
                vertices_line.clear()

            self.vertices_count += 0
            index_char += 1
        self.position = position_initial

        if self.centered:
            self.center_text_horizontal(box_center_horizontal, vertices_line)
            vertices += vertices_line
            self.center_text_vertically(vertices, vertices_line)
        else:
            vertices += vertices_line

        return np.array(
            vertices,
            dtype=np.float32
        )

    def center_text_vertically(self, vertices, vertices_line):
        box_bottom = self.position.y - (self.scale[1] * 2)
        box_center_vertical = (self.position.y + box_bottom) / 2
        text_center_vertical = (vertices[1] + vertices_line[-3]) / 2
        shift = box_center_vertical - text_center_vertical
        for index, value in enumerate(vertices):
            if index % 4 == 1:
                vertices[index] += shift

    def center_text_horizontal(self, box_center_horizontal, vertices_line):
        text_center_horizontal = (vertices_line[0] + vertices_line[-8]) / 2.0
        shift = box_center_horizontal - text_center_horizontal
        for index, value in enumerate(vertices_line):
            if index % 4 == 0:
                vertices_line[index] += shift

    def update_text(self, text=None, color=None):
        if text:
            self.text = text
        if color:
            self.color = glm.vec4(color)
        self.vertices = self.generate_vertices()
        self.clean_up()
        self.buffer_setup()

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

        glUniform4fv(self.color_loc, 1, glm.value_ptr(self.color))

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_DEPTH_TEST)
        # glDrawArrays(GL_TRIANGLE_STRIP, 0, self.vertices_count)
        # glDrawArrays(GL_TRIANGLES, 0, self.vertices_count)
        glDrawArrays(GL_TRIANGLES, 0, int(len(self.vertices) / 3))
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)