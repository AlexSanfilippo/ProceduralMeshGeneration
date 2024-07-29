"""
17th July 2024
Following LearnOpenGL.com's tutorial on skyboxes and cubemaps
"""
from OpenGL.GL import *
from TextureLoader import load_cube_map
import numpy as np
import glm
from OpenGL.GL.shaders import compileProgram, compileShader

vertex_src = """
#version 330
layout (location = 0) in vec3 aPos;

out vec3 TexCoords;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    TexCoords = aPos;
    vec4 pos = projection * view * vec4(aPos, 1.0);
    gl_Position = pos;
}  
"""

fragment_src = """
#version 330
out vec4 FragColor;

in vec3 TexCoords;

uniform samplerCube skybox;

void main()
{    
    FragColor = texture(skybox, TexCoords);
}

"""
class Skybox:
    """
    Create a skybox using a cubemap
    textures: a list of 6 textures paths in the order:
        right, left, top, bottom, front, back
    """

    def __init__(self, texture_paths, scale=1000):

        self.vertices = np.array(
            [
                # positions
                -1.0,  1.0, -1.0,
                -1.0, -1.0, -1.0,
                 1.0, -1.0, -1.0,
                 1.0, -1.0, -1.0,
                 1.0,  1.0, -1.0,
                -1.0,  1.0, -1.0,

                -1.0, -1.0,  1.0,
                -1.0, -1.0, -1.0,
                -1.0,  1.0, -1.0,
                -1.0,  1.0, -1.0,
                -1.0,  1.0,  1.0,
                -1.0, -1.0,  1.0,

                 1.0, -1.0, -1.0,
                 1.0, -1.0,  1.0,
                 1.0,  1.0,  1.0,
                 1.0,  1.0,  1.0,
                 1.0,  1.0, -1.0,
                 1.0, -1.0, -1.0,

                -1.0, -1.0,  1.0,
                -1.0,  1.0,  1.0,
                 1.0,  1.0,  1.0,
                 1.0,  1.0,  1.0,
                 1.0, -1.0,  1.0,
                -1.0, -1.0,  1.0,

                -1.0,  1.0, -1.0,
                 1.0,  1.0, -1.0,
                 1.0,  1.0,  1.0,
                 1.0,  1.0,  1.0,
                -1.0,  1.0,  1.0,
                -1.0,  1.0, -1.0,

                -1.0, -1.0, -1.0,
                -1.0, -1.0,  1.0,
                 1.0, -1.0, -1.0,
                 1.0, -1.0, -1.0,
                -1.0, -1.0,  1.0,
                 1.0, -1.0,  1.0,
            ],
            dtype=np.float32
        ) * scale
        self.textures = self.create_cube_map(texture_paths)
        self.shader = self.create_shader()
        self.setup_buffers()

    def create_shader(self):
        shader = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER)
        )
        return shader

    def setup_buffers(self):
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        # position vertices (vertex attribute)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 3, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

    def create_cube_map(self, texture_paths):
        textures = glGenTextures(1)
        paths = texture_paths
        load_cube_map(paths, textures)
        return textures

    def draw(self, view, projection):
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.textures)

        proj_loc = glGetUniformLocation(self.shader, "projection")
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

        #remove translation
        view_no_transform = view.copy()
        view = np.array(glm.mat4x4(glm.mat3x3(glm.mat4x4(view_no_transform.transpose()))).to_list())

        #view matrix to shader
        view_loc = glGetUniformLocation(self.shader, "view")
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

        glDrawArrays(GL_TRIANGLES, 0, int(len(self.vertices) / 3))
