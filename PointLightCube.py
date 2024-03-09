"""
    7th August, 2023
    Brief: A class for a cube that emits a point light
"""

import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

import pyrr
import numpy as np


#Let's have it contain its own shader

vertex_src = """
# version 330

layout(location = 0) in vec3 a_position;

uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;


void main()
{
    gl_Position = projection * view * model * vec4(a_position, 1.0);
    //gl_Position = projection * view * vec4(a_position, 1.0);
    //gl_Position = vec4(a_position, 1.0); //SAVE FOR GUI LATER
  
}
"""

fragment_src = """
# version 330

out vec4 out_color;

uniform vec3 light_color;


void main()
{
    out_color = vec4(light_color,1.f);
}
"""


class PointLightCube:
    def __init__(self,
                 pos=[0.0, 0.0, 0.0],
                 ambient=[1.0, 1.0, 1.0],
                 diffuse=[1.0, 1.0, 1.0],
                 specular=[1.0, 1.0, 1.0],
                 scale=1.0,
                 constant=1.0,
                 linear=0.014,
                 quadratic=0.00007
                 ):
        self.pos = pos
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.scale = scale

        # falloff
        self.constant = constant
        self.linear = linear
        self.quadratic = quadratic

        self.vertices = np.array([-0.5, -0.5, 0.5,
                    0.5, -0.5, 0.5,
                    0.5, 0.5, 0.5,
                    -0.5, 0.5, 0.5,

                    -0.5, -0.5, -0.5,
                    0.5, -0.5, -0.5,
                    0.5, 0.5, -0.5,
                    -0.5, 0.5, -0.5,

                    0.5, -0.5, -0.5,
                    0.5, 0.5, -0.5,
                    0.5, 0.5, 0.5,
                    0.5, -0.5, 0.5,

                    -0.5, 0.5, -0.5,
                    -0.5, -0.5, -0.5,
                    -0.5, -0.5, 0.5,
                    -0.5, 0.5, 0.5,

                    -0.5, -0.5, -0.5,
                    0.5, -0.5, -0.5,
                    0.5, -0.5, 0.5,
                    -0.5, -0.5, 0.5,

                    0.5, 0.5, -0.5,
                    -0.5, 0.5, -0.5,
                    -0.5, 0.5, 0.5,
                    0.5, 0.5, 0.5], dtype=np.float32) * scale

        self.indices = np.array([0, 1, 2, 2, 3, 0,
                   4, 5, 6, 6, 7, 4,
                   8, 9, 10, 10, 11, 8,
                   12, 13, 14, 14, 15, 12,
                   16, 17, 18, 18, 19, 16,
                   20, 21, 22, 22, 23, 20], dtype=np.uint32)

        # cube VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        # Element Buffer Object
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)


        # cube vertices (vertex attribute)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 3, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        self.shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), \
                                     compileShader(fragment_src, GL_FRAGMENT_SHADER))

        self.light_color_loc = glGetUniformLocation(self.shader, "light_color")

        self.model_loc = glGetUniformLocation(self.shader, "model")
        self.proj_loc = glGetUniformLocation(self.shader, "projection")
        self.view_loc = glGetUniformLocation(self.shader, "view")

        self.projection = pyrr.matrix44.create_perspective_projection_matrix(45, 1280 / 720, 0.1, 1000)
        # glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, self.projection)

        self.translation = pyrr.matrix44.create_from_translation(pyrr.Vector3(pos))
        self.rotation = pyrr.matrix44.create_from_quaternion([1.0, 1.0, 1.0, 1.0])

    def draw(self,view):
        glBindVertexArray(self.VAO)
        glUseProgram(self.shader)
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, view)
        # glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, self.pos)
        model_mat = pyrr.matrix44.create_identity()
        model_mat = pyrr.matrix44.multiply(self.rotation, self.translation) #.translation * self.rotation
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, model_mat)
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, self.projection)
        glUniform3fv(self.light_color_loc, 1, self.diffuse)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
        # glDrawArrays(GL_TRIANGLES, 0, 36);

    def set_ambient(self, ambient):
        self.ambient = ambient

    def set_diffuse(self, diffuse):
        self.diffuse = diffuse

    def set_specular(self, specular):
        self.specular = specular

    def get_ambient(self):
        return self.ambient

    def get_diffuse(self):
        return self.diffuse

    def get_specular(self):
        return self.specular

    def get_pos(self):
        return self.pos

    def get_constant(self):
        return self.constant

    def get_linear(self):
        return self.linear

    def get_quadratic(self):
        return self.quadratic

    def set_pos(self, pos):
        self.pos = pos
        self.translation = pyrr.matrix44.create_from_translation(pyrr.Vector3(self.pos))

    # rotate the PointLightCube, given a quaternion from bullet
    def set_orientation(self, orientation):
        # cube_pos = pyrr.matrix44.create_from_translation(pyrr.Vector3([6, 4, 0]))

        #opt A
        # rot_y = pyrr.Matrix44.from_y_rotation(orientation)
        #opt B
        # rot_y = pyrr.matrix44.create_from_inverse_of_quaternion(orientation)
        # opt C
        rot_y = pyrr.Matrix44.from_quaternion(orientation)
        #pass to shader
        self.rotation = rot_y #pyrr.matrix44.multiply(rot_y, self.translation)

    # TP: make sure we can rotate the object this way
    def rotate_cube(self):
        rot_y = pyrr.Matrix44.from_y_rotation(0.8 * glfw.get_time())
        self.rotation = rot_y #pyrr.matrix44.multiply(rot_y, self.translation)

