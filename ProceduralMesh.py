"""
1st December, 2023

Classes for creation of basic proc-gen shapes.  Shapes to create will be:

-[V]Triangle
-[V]Quad
-[V]Cube

-[V]Dynamic Size on cube using Dimensions [x,y,z]
-[V]Proper control over scaling, rotation, and transposition
    -goal:rotating cube
-[V]hard-code normals on cube
-[V]refactor
    -put draw, init, etc. into PrimativeMesh class
        -new functions to set vertices and indices
            -these will be overridden in the children


-----Stop here for proc jam, continue later
-[V]Proper normal vector calculation
    -grab/rewrite my old code from primatives.h
    -properly lit rotating cube

-[V]N-Sided Polygon
-[V]N-sided Prism
-[]N-sided pyramid
-[]Sphere
-[]Ellipsoid
-[]Torus

-for now, need some basic shapes for procjam 2023.  Will start with triangle, quad, and cube
-for all meshes, we will assume they have
    -position, texture coordinate, and normal vector as vertex info
    -specular and diffuse textures
    -use indexed drawing (indices and vertices)
"""

import glfw
import glm
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from TextureLoader import load_texture

import pyrr
import numpy as np
import math
from math import sin, cos, pi, sqrt, floor, ceil
from random import random
import random as r

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

#define NR_POINT_LIGHTS 1

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
    //result += CalcSpotLight(spot_light, norm, frag_pos, view_dir);    

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

class PrimativeMesh():
    """
    abstract parent of all primative meshes
        -draws quad by default
    """
    def __init__(self,
                 shader,
                 # material properties
                 diffuse,
                 specular,
                 shininess=32.0,
                 # mesh properties
                 dimensions=(5.0, 5.0),
                 position=(0.0, 0.0, 0.0),
                 rotation_magnitude=(0, 0, 0),
                 rotation_axis=(0.0, 0.0, 1.0),
                 scale=(1.0, 1.0, 1.0),
                 ):
        self.shader = shader
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.position = glm.vec3(position)
        self.rotation_magnitude = glm.vec3(rotation_magnitude)
        self.rotation_axis = glm.vec3(rotation_axis)
        self.scale = glm.vec3(scale)
        self.dimensions = dimensions
        self.vertices = self.generate_vertices()
        self.buffer_setup()

    def buffer_setup(self):
        # quad VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        # quad position vertices (vertex attribute)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # quad texture coords
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 8, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        # quad normals
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 8, ctypes.c_void_p(20))
        glEnableVertexAttribArray(2)

        self.model_loc = glGetUniformLocation(self.shader, "model")

    def generate_vertices(self):
        return np.array([
            1.0 * self.dimensions[0], 1.0 * self.dimensions[1], 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
            -1.0 * self.dimensions[0], 1.0 * self.dimensions[1], 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
            -1.0 * self.dimensions[0], -1.0 * self.dimensions[1], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            1.0 * self.dimensions[0], 1.0 * self.dimensions[1], 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
            -1.0 * self.dimensions[0], -1.0 * self.dimensions[1], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            1.0 * self.dimensions[0], -1.0 * self.dimensions[1], 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32
        )

    def draw(self, view):
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.diffuse)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.specular)

        # rotate, translate, and scale
        model = glm.mat4(1.0)
        model = glm.translate(model, self.position)
        #todo: this rotation smells fishy
        model = glm.rotate(model, self.rotation_magnitude.x, self.rotation_axis)
        model = glm.scale(model, self.scale)
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, glm.value_ptr(model))

        glDrawArrays(GL_TRIANGLES, 0, int(len(self.vertices)/3))

    def set_diffuse(self, diffuse):
        self.diffuse = diffuse

    def set_specular(self, specular):
        self.specular = specular

    def get_diffuse(self):
        return self.diffuse

    def get_specular(self):
        return self.specular

    def get_position(self):
        return self.position

    def set_position(self, position):
        self.position = position
        # self.translation = pyrr.matrix44.create_from_translation(pyrr.Vector3(self.position))

    def set_rotation_magnitude(self, magnitude):
        self.rotation_magnitude = magnitude

    def rotate_over_time(self, speed=0.5, axis=1):
        """
        my_model.rotate_over_time(speed=0.1)
        :param speed: how fast to spin
        :param axis: x,y,z == 0,1,2
        :return: None
        """
        self.rotation_magnitude[axis] = speed * glfw.get_time()

    #todo: This function already exists in Face and is used nowhere.
    # def generate_texture_coordinates_polygonal(self, sides, reverse=True):
    #     """
    #     A new means of generating texture coordinates for any (regular) polygon.
    #     Even works for polygons not on XZ plane at origin
    #     :return: a list of glm.vec2() coordinates, one for each point of shape
    #     """
    #     angle_delta = -2.0 * pi / sides
    #     angle_initial = (pi / sides) - (pi / 2.0)
    #     radius = 0.5
    #     if sides == 4:
    #         radius = 1.0 / math.sqrt(2.0)
    #         angle_initial = (pi / sides) + angle_delta
    #     texture_coordinates = []
    #     for index_point in range(sides):
    #         coordinate = glm.vec2([0.0, 0.0])
    #         coordinate.x = (cos(angle_initial - angle_delta * index_point) * radius) + 0.5
    #         coordinate.y = (sin(angle_initial - angle_delta * index_point) * radius) + 0.5
    #         texture_coordinates.append(coordinate)
    #     if reverse:
    #         texture_coordinates.reverse()
    #     return texture_coordinates

    def extrude_from_other_face(self, other, direction = [0.0, 1.0, 0.0], distance=1.0, flip_base=True, radius=1.0):
        """
        just copy outer vertices, then run outer_Vertices_to_vertices()
        Very different from (and likely outmoding) Face's method of same name
        :param other: starting face
        :param direction: normal vector to extrude in direcion of
        :param distance: multiply direction by for magnitude of extrude
        :param flip_base: turn starting face around
        :param radius: (only if position == world origin) scale face
        :return: a new face in a list
        """
        face_extruded = Face()
        face_extruded.outer_vertices = other.outer_vertices.copy()
        face_extruded.calculate_sides()
        face_extruded.vertices = self.outer_vertices_to_vertices(face=face_extruded, number_of_sides=face_extruded.sides, reverse_texture_coords=True)
        face_extruded.normal = face_extruded.calculate_normal()
        if flip_base:
            pass
        return face_extruded

    def outer_vertices_to_vertices(self, face, number_of_sides, reverse_texture_coords=False):
        """
        Generates the real vertices (with normals, texture coords) of a face
        :param reverse_texture_coords: reverse order of list of position vertices
        :param face: the face we want to generate vertices for.  must have its outer vertices already
        :param number_of_sides: eg, 4 for square, 5 for pentagon
        :return: the vertices to be drawn by opengl
        """

        # texture_coordinates = face.generate_texture_coordinates_polygon()
        texture_coordinates = face.generate_texture_coordinates_polygon()
        if reverse_texture_coords:
            texture_coordinates.reverse()
        vertices = []
        texture_index = 1
        for i in range(int(number_of_sides) - 2):
            triangle_vertices = [Vertex(), Vertex(), Vertex()]

            triangle_vertices[0].positions = glm.vec3([
                face.outer_vertices[0],
                face.outer_vertices[1],
                face.outer_vertices[2]
            ])
            triangle_vertices[0].texture_coordinates = glm.vec2([
                texture_coordinates[0].x,
                texture_coordinates[0].y
            ])
            triangle_vertices[0].normals = glm.vec3([0.0, 1.0, 0.0])

            triangle_vertices[1].positions = glm.vec3([
                face.outer_vertices[3 * i + 0 + 3 * 1],
                face.outer_vertices[3 * i + 1 + 3 * 1],
                face.outer_vertices[3 * i + 2 + 3 * 1]
            ])
            triangle_vertices[1].texture_coordinates = glm.vec2([
                texture_coordinates[texture_index].x,
                texture_coordinates[texture_index].y
            ])
            triangle_vertices[1].normals = glm.vec3([0.0, 1.0, 0.0])

            triangle_vertices[2].positions = glm.vec3([
                face.outer_vertices[3 * i + 0 + 3 * 2],
                face.outer_vertices[3 * i + 1 + 3 * 2],
                face.outer_vertices[3 * i + 2 + 3 * 2]
            ])
            triangle_vertices[2].texture_coordinates = glm.vec2([
                texture_coordinates[texture_index + 1].x,
                texture_coordinates[texture_index + 1].y
            ])
            triangle_vertices[2].normals = glm.vec3([0.0, 1.0, 0.0])
            for vertex in triangle_vertices:
                vertex.normals = calculate_normal([
                    triangle_vertices[0].positions,
                    triangle_vertices[1].positions,
                    triangle_vertices[2].positions
                ])
            vertices += triangle_vertices
            texture_index += 1
        return vertices

    def _generate_polygonal_face(
            self,
            number_of_sides,
            radius=1.0,
            transform_x=1.0,
            transform_z=1.0,
            offset=(0.0, -1.0, 0.0)
    ):
        face = Face()
        face.outer_vertices = self.generate_outer_vertices(
            number_of_sides=number_of_sides,
            initial_angle=pi / number_of_sides,
            radius=radius,
            transform_x=transform_x,
            transform_z=transform_z,
        )
        face.offset_outer_vertices(offset=glm.vec3(offset))
        face.vertices = face.outer_vertices_to_vertices(reverse_texture_coordinates=True)
        return face

    def _generate_irregular_polygonal_face(
            self,
            number_of_sides,
            radii=1.0,
            transform_x=1.0,
            transform_z=1.0,
            offset=(0.0, -1.0, 0.0)
    ):
        face = Face()
        face.outer_vertices = self.generate_outer_vertices(
            number_of_sides=number_of_sides,
            initial_angle=pi / number_of_sides,
            radius=radii,
            transform_x=transform_x,
            transform_z=transform_z,
        )
        face.offset_outer_vertices(offset=glm.vec3(offset))
        face.vertices = face.outer_vertices_to_vertices(reverse_texture_coordinates=True)
        return face



    def generate_outer_vertices(
            self,
            number_of_sides=4,
            initial_angle=None,
            radius=1.0,
            transform_x=1.0,
            transform_z=1.0,
    ):
        """
        generates the vertices of a simple polygon-no repeats
        used for excuding and stitching, not for drawing
        """
        outer_vertices = []
        angle_of_rotation = 2 * pi / number_of_sides
        if initial_angle == None:
            initial_angle = pi / number_of_sides
        if type(radius) is list:
            for i in range(1, int(number_of_sides) + 1):
                x_coord = cos(initial_angle + angle_of_rotation * -i) * radius[i-1] * transform_x
                z_coord = sin(initial_angle + angle_of_rotation * -i) * radius[i-1] * transform_z
                outer_vertices += [x_coord, 0.0, z_coord]
        else:
            for i in range(1, int(number_of_sides) + 1):
                x_coord = cos(initial_angle + angle_of_rotation * -i) * radius * transform_x
                z_coord = sin(initial_angle + angle_of_rotation * -i) * radius * transform_z
                outer_vertices += [x_coord, 0.0, z_coord]
        return outer_vertices

    def bevel_cut(self, original_face, bevel_depths=[-0.2, -0.4], border_sizes=[0.6, 0.0], depth=2, direction=False):
        """
        Cuts an n-sided hole, offsets a new face, and stitches to create a connected-depression
        :bevel_depths: how much to push the excude in/out (negative for in)
        :border_sizes: elm of (0,1), 0 means no border, aka, cut is same size as original polygonal face
        :return: list of new faces. new normal-facing face (bottom face) is at index [:-(nsides + 1)]

        TODO:
            -Refactoring for neatness/clarity
            -wrong texture coords for irregular convex polygons
        """

        if depth == 0:
            return [original_face]

        original_face.calculate_radius()
        radius = original_face.radius
        updated_faces = []

        test_cut_face = Face()
        original_outer_vertices = list_to_vec3_list(original_face.outer_vertices)
        for i in range(depth):
            #1. cut a similar polygon from the starting face
            cut_outer_vertices = []
            for point in original_outer_vertices:
                mean_outer_point = calculate_mean_point(original_outer_vertices)
                border_direction = glm.normalize(mean_outer_point - point)
                border_diagonal_length = radius*border_sizes[i]
                cut_point = point + border_direction*border_diagonal_length
                cut_outer_vertices.append(cut_point)

            #2. build new face uses new outer vertices
            test_cut_face.outer_vertices = vec3_list_to_list(cut_outer_vertices)
            test_cut_face.calculate_sides()
            test_cut_face.calculate_outer_vertices_as_vec3()
            if direction == False:
                face_normal = calculate_normal([
                    test_cut_face.outer_vertices_vec3[0],
                    test_cut_face.outer_vertices_vec3[1],
                    test_cut_face.outer_vertices_vec3[2]]
                )
            else:
                face_normal = direction
            test_cut_face.offset_outer_vertices(offset=face_normal*bevel_depths[i])
            test_cut_face.vertices = test_cut_face.outer_vertices_to_vertices(
                                reverse_texture_coordinates=True,
                                regular=False,
            )
            if i == depth - 1:
                updated_faces.append(test_cut_face)
            #3. stitch old faces to new face
            updated_faces += self.stitch_faces(original_face, test_cut_face, test_cut_face.sides)

            #4. new face becomes "original" in preperation for next loop
            original_face.outer_vertices = test_cut_face.outer_vertices
            original_face.vertices = test_cut_face.vertices
            original_face.calculate_radius()
            radius = original_face.radius
            original_outer_vertices = list_to_vec3_list(test_cut_face.outer_vertices.copy())

        return updated_faces

    def bevel_polygon_corner(self, face, subject_vertex_index=0, bevel_ratio = 0.5):
        """
        Takes in a face, transforms outer vertices of polygon by beveling a corner
            -use case: base polygon before excude
        :subject_vertex_index: index of outer vertex to bevel
        :bevel_ratio: domain [0,1].
        :return: the modified face
        """
        older_outer_vertices = list_to_vec3_list(face.outer_vertices.copy())
        subject_vertex = older_outer_vertices[subject_vertex_index]
        number_of_sides = len(older_outer_vertices)
        forward_vertex = older_outer_vertices[get_next_index(number_of_sides, subject_vertex_index)]
        backward_vertex = older_outer_vertices[get_previous_index(number_of_sides, subject_vertex_index)]
        forward_new_point = subject_vertex - bevel_ratio*(subject_vertex - forward_vertex)
        backward_new_point = subject_vertex - bevel_ratio*(subject_vertex - backward_vertex)

        updated_outer_vertices = older_outer_vertices.copy()
        del updated_outer_vertices[subject_vertex_index]
        updated_outer_vertices.insert(subject_vertex_index, forward_new_point)
        updated_outer_vertices.insert(subject_vertex_index, backward_new_point)
        face.outer_vertices = vec3_list_to_list(updated_outer_vertices)
        self.sides += 1
        return face

    def pyrimidize_face(self, original_face, center_point_offset=0.0):
        """
        Changes how an N-sided face is divided into triangles.  Using a center point, and two corner points,
        N triangles are created for the faces.
        :param original_face: face to transform into N new faces
        :param center_point_offset:
        :return: N faces
        """
        # get normal of the face
        original_face.calculate_outer_vertices_as_vec3()
        original_face.normal = original_face.calculate_normal()
        original_normal = original_face.normal
        # calculate center of original face
        original_outer_vertices_as_vec3s = original_face.outer_vertices_vec3
        center_point = calculate_mean_point(original_outer_vertices_as_vec3s)
        # adjust center by the normal and center_point_offset
        center_point += original_normal * center_point_offset
        # construct the N new triangles outer vertices
        original_face.calculate_sides()
        num_sides = original_face.sides
        original_outer_vertices_as_vec3s = original_face.outer_vertices_vec3

        new_triangles_outer_vertices=[]
        for index, original_outer_vertex in enumerate(original_outer_vertices_as_vec3s):
            triangle_vertices = []
            next_index = index + 1
            if index == num_sides - 1:
                next_index = 0
            triangle_vertices += [original_outer_vertices_as_vec3s[next_index], center_point, original_outer_vertex]
            new_triangles_outer_vertices.append(triangle_vertices)

        # create N new faces (using outer vertices)
        triangle_faces = []
        for index in range(num_sides):
            triangle_faces.append(Face())
            # -convert vec3 outers into list outers
            triangle_faces[index].outer_vertices = vec3_list_to_list(new_triangles_outer_vertices[index])
            triangle_faces[index].outer_vertices_vec3 = new_triangles_outer_vertices[index]
            # generate vertices of N new faces.
            # triangle_faces[index].vertices = triangle_faces[index].outer_vertices
            triangle_faces[index].vertices = triangle_faces[index].outer_vertices_to_vertices()
            triangle_faces[index].apply_hardset_triangle_texture_coords()
        # return the N new quad faces
        return triangle_faces


        #make n new triangular faces

    def divide_face_into_quads(self, original_face, center_point_offset=0.0):
        """
        subdivide a face into quads.  each new quad has as its vertices 2 midpoints, 1 corner, and the center of
        the original face.  Only works on quad faces?
        :param original_face: a face object
        :param center_point_offset: move center along face normal
        :return: 4 new faces
        """

        #get normal of the face
        original_face.calculate_outer_vertices_as_vec3()
        original_face.normal = original_face.calculate_normal()
        original_normal = original_face.normal
        #calculate center of original face
        original_outer_vertices_as_vec3s = original_face.outer_vertices_vec3
        center_point = calculate_mean_point(original_outer_vertices_as_vec3s)
        #adjust center by the normal and center_point_offset
        center_point += original_normal * center_point_offset
        #find the N midpoints (N = number of sides of face polygon)
        original_face.calculate_sides()
        num_sides = original_face.sides
        midpoints = [glm.vec3(0.0, 0.0, 0.0)] * num_sides
        for index in range(num_sides):
            overflowing_index = 1
            if index == num_sides - 1:
                overflowing_index = -index
            midpoints[index] = calculate_mean_point(
                [original_outer_vertices_as_vec3s[index],
                 original_outer_vertices_as_vec3s[index + overflowing_index]
                 ]
            )
        #construct the N quads outer vertices.
        all_quads_outer_vertices = []
        for index, outer_point in enumerate(original_outer_vertices_as_vec3s):
            current_quad_outer_vertices = []
            if index > 0:
                backwards_midpoint_index = index - 1
            else:
                backwards_midpoint_index = num_sides - 1

            current_quad_outer_vertices.append(outer_point)
            current_quad_outer_vertices.append(midpoints[index])
            current_quad_outer_vertices.append(center_point)
            current_quad_outer_vertices.append(midpoints[backwards_midpoint_index])
            current_quad_outer_vertices = rotate_list(current_list=current_quad_outer_vertices, steps=-index)
            all_quads_outer_vertices.append(current_quad_outer_vertices)

        #create N new faces (using outer vertices)
        quad_faces = []
        for index in range(num_sides):
            quad_faces.append(Face())
            #-convert vec3 outers into list outers
            quad_faces[index].outer_vertices = vec3_list_to_list(all_quads_outer_vertices[index])
            quad_faces[index].outer_vertices_vec3 = all_quads_outer_vertices[index]
            # generate vertices of N new faces.
            quad_faces[index].vertices = quad_faces[index].outer_vertices_to_vertices()
            # (option) fix texture coords of N new faces
            # quad_faces[index].apply_hardset_quad_texture_coords(sides=4)

        #return the N new quad faces
        return quad_faces

    def bezier_cut(self, face, intervals=0, offset=0):
        """
        Constructs a bevel curve from the outer vertices of a quad-face, and uses the curve to cut a hole
        in the face, creating two new faces
        :param face:
        :param iterations:
        :return: list of faces
        """
        face.calculate_outer_vertices_as_vec3()
        original_face_normal = face.calculate_normal()
        from collections import deque
        d = deque(face.outer_vertices_vec3)
        d.rotate(2)
        d.reverse()
        control_points = list(d)
        bezier_points = bezier_cubic(points=control_points, intervals=intervals)
        vertices_border = []
        outer_vertices_cut = []

        even_odd_index = 0
        if intervals % 2 == 0:
            even_odd_index = 1
            
        #first triangle
        vertices_border.append(face.outer_vertices_vec3[0])
        vertices_border.append(face.outer_vertices_vec3[1])
        vertices_border.append(bezier_points[1])

        midpoint = len(bezier_points)/2
        midpoint_indices = [math.floor(midpoint), math.ceil(midpoint)]
        # for index, bezier_point in enumerate(bezier_points[1:midpoint_indices[1]+1]):
        for index, bezier_point in enumerate(bezier_points[1:midpoint_indices[even_odd_index]]):
            vertices_border.append(bezier_point)
            vertices_border.append(bezier_points[index + 2])
            vertices_border.append(face.outer_vertices_vec3[0])
        #keystone_triangle (even vertices case)
        vertices_border.append(bezier_points[midpoint_indices[even_odd_index]])
        vertices_border.append(face.outer_vertices_vec3[3])
        vertices_border.append(face.outer_vertices_vec3[0])
        for index, bezier_point in enumerate(bezier_points[midpoint_indices[even_odd_index]:-1]):
            vertices_border.append(bezier_point)
            vertices_border.append(bezier_points[index + midpoint_indices[even_odd_index] + 1])
            vertices_border.append(face.outer_vertices_vec3[3])
        proper_vertices = []
        for point in vertices_border:
            current_vertex = Vertex()
            current_vertex.positions = point
            current_vertex.texture_coordinates = calculate_texture_coordinates_from_control_points(
                point=point,
                control_points=control_points
            )
            current_vertex.normals = original_face_normal
            proper_vertices.append(current_vertex)

        face.vertices = proper_vertices

        """Now do the cut's vertices"""
        #cut saved into a new face
        face_cut = Face()
        face_cut.outer_vertices = vec3_list_to_list([point - original_face_normal*offset for point in bezier_points])
        midpoint_base = calculate_mean_point([bezier_points[0], bezier_points[intervals + 1]])
        vertices_cut = []
        for index, bezier_point in enumerate(bezier_points[:intervals+1]):
            vertices_cut.append(midpoint_base - original_face_normal*offset)
            vertices_cut.append(bezier_points[index+1] - original_face_normal*offset)
            vertices_cut.append(bezier_point - original_face_normal*offset)
        vertices_cut_proper = []
        for point in vertices_cut:
            current_vertex = Vertex()
            current_vertex.positions = point
            current_vertex.texture_coordinates = calculate_texture_coordinates_from_control_points(
                point=point,
                control_points=control_points
            )
            current_vertex.normals = original_face_normal
            vertices_cut_proper.append(current_vertex)

        face_cut.vertices = vertices_cut_proper
        faces_inner_arch = []
        if offset != 0:
            face_cut_pre_offset = Face()
            face_cut_pre_offset.outer_vertices = vec3_list_to_list(bezier_points)
            return [face, face_cut] + self.stitch_faces(face_cut, face_cut_pre_offset, intervals + 2)
        return [face, face_cut]

    def stitch_faces(self, face_a, face_b, number_of_faces):
        """
        takes two (geometrically same) Faces and forms a prism-like mesh by connecting them
        with quadrilaterals
        :returns: a list of faces
        """
        faces = []
        for side in range(int(number_of_faces)):
            face = Face()

            if side == number_of_faces - 1:
                face.outer_vertices = [
                    face_b.outer_vertices[side * 3 + 0], face_b.outer_vertices[side * 3 + 1], face_b.outer_vertices[side * 3 + 2],
                    face_a.outer_vertices[side * 3 + 0], face_a.outer_vertices[side * 3 + 1], face_a.outer_vertices[side * 3 + 2],
                    face_a.outer_vertices[0 * 3 + 0], face_a.outer_vertices[0 * 3 + 1],face_a.outer_vertices[0 * 3 + 2],
                    face_b.outer_vertices[0 * 3 + 0], face_b.outer_vertices[0 * 3 + 1],face_b.outer_vertices[0 * 3 + 2],
                ]
            else:
                face.outer_vertices = [
                    face_b.outer_vertices[side * 3 + 0], face_b.outer_vertices[side * 3 + 1], face_b.outer_vertices[side * 3 + 2],
                    face_a.outer_vertices[side * 3 + 0], face_a.outer_vertices[side * 3 + 1], face_a.outer_vertices[side * 3 + 2],
                    face_a.outer_vertices[side * 3 + 3], face_a.outer_vertices[side * 3 + 4], face_a.outer_vertices[side * 3 + 5],
                    face_b.outer_vertices[side * 3 + 3], face_b.outer_vertices[side * 3 + 4], face_b.outer_vertices[side * 3 + 5],
                ]
            # face.vertices = self.outer_vertices_to_vertices(face, 4)
            face.vertices = face.outer_vertices_to_vertices(reverse_texture_coordinates=True)

            faces.append(face)

        return faces

    def subdivide_hip_roof(self, face, start_point=[0.5, 0.2], end_point=[0.5, 0.8], vertical_offset=0.0):
        """
        divides (quad) face into 2 quads and 2 triangle faces, like the triangular "hip" roof of a house
        todo:fix texture coords
        :return: list of faces: two square, two triangular
        """
        original_outer_vertices = face.outer_vertices.copy()
        operatable_outer_vertices = original_outer_vertices.copy()
        operatable_outer_vertices_vec3 = list_to_vec3_list(operatable_outer_vertices)
        # make sure original face has all required attributes
        face.calculate_outer_vertices_as_vec3()
        face.normal = face.calculate_normal()
        original_normal = face.normal
        original_outer_vertices_as_vec3s = face.outer_vertices_vec3
        face.calculate_sides()

        # calculate the start and end points of the center line
        center_line_start_point = (operatable_outer_vertices_vec3[1]
            + start_point[0] * (operatable_outer_vertices_vec3[0] - operatable_outer_vertices_vec3[1])
            + start_point[1] * (operatable_outer_vertices_vec3[2] - operatable_outer_vertices_vec3[1])
            + vertical_offset * face.normal)
        center_line_end_point = (operatable_outer_vertices_vec3[1]
            + end_point[0]*(operatable_outer_vertices_vec3[0] - operatable_outer_vertices_vec3[1])
            + end_point[1]*(operatable_outer_vertices_vec3[2] - operatable_outer_vertices_vec3[1])
            + vertical_offset * face.normal)

        # generate outer vertices of new quad faces
        roof_faces = []
        quad_base_points = [
            [0, 3, center_line_start_point, center_line_end_point],
            [2, 1, center_line_end_point, center_line_start_point]
        ]
        for base_points in quad_base_points:
            current_outer_vertices = []
            current_outer_vertices.append(original_outer_vertices_as_vec3s[base_points[0]])
            current_outer_vertices.append(base_points[2])
            current_outer_vertices.append(base_points[3])
            current_outer_vertices.append(original_outer_vertices_as_vec3s[base_points[1]])
            new_face = Face()
            new_face.outer_vertices_vec3=current_outer_vertices
            new_face.outer_vertices=vec3_list_to_list(current_outer_vertices)
            new_face.vertices = new_face.outer_vertices_to_vertices()
            new_face.apply_hardset_quad_texture_coords()
            roof_faces.append(new_face)

        triangle_base_points = [[1, 0, center_line_start_point], [3, 2, center_line_end_point]]
        for i in range(2):
            current_outer_vertices = []
            current_outer_vertices.append(original_outer_vertices_as_vec3s[triangle_base_points[i][0]])
            current_outer_vertices.append(triangle_base_points[i][2])
            current_outer_vertices.append(original_outer_vertices_as_vec3s[triangle_base_points[i][1]])
            new_face = Face()
            new_face.outer_vertices_vec3=current_outer_vertices
            new_face.outer_vertices=vec3_list_to_list(current_outer_vertices)
            new_face.vertices = new_face.outer_vertices_to_vertices()
            #todo:fix these triangle faces outer vertices
            new_face.apply_hardset_triangle_texture_coords()
            roof_faces.append(new_face)

        return roof_faces

    def subdivide_quad_lengthwise(self, face, subdivisions=2, widthwise=False):
        """
        Takes in a quadrillateral face and divides it into subfaces along the length of the face
        :return: list of faces

        todo: the widthwise option rotates and distorts the texture
            -need a way to do widthwise that does not rely on rotating the list naively
        """
        original_outer_vertices = face.outer_vertices.copy()
        operatable_outer_vertices = original_outer_vertices.copy()
        if widthwise:
            operatable_outer_vertices = rotate_list(current_list=operatable_outer_vertices, steps=3)
            #todo: rotate back at the end -necc?
        if subdivisions in (0,1):
            return [face]
        
        updated_faces = []
        #make sure original face has all required attributes
        face.calculate_outer_vertices_as_vec3()
        face.normal = face.calculate_normal()
        original_normal = face.normal
        original_outer_vertices_as_vec3s = face.outer_vertices_vec3
        face.calculate_sides()
        
        #find the N-subdivisions points along the length sides
        operatable_outer_vertices_vec3 = list_to_vec3_list(operatable_outer_vertices)
        side_top = [operatable_outer_vertices_vec3[0], operatable_outer_vertices_vec3[3]]
        side_bottom = [operatable_outer_vertices_vec3[1], operatable_outer_vertices_vec3[2]]
        outer_vertices_top = bezier_linear(points=side_top, intervals=subdivisions)
        outer_vertices_bottom = bezier_linear(points=side_bottom, intervals=subdivisions)


        #generate outer vertices of new faces
        subdivision_faces = []
        subdivision_index = 0
        while len(subdivision_faces) < subdivisions:
            subdivision_face = Face()
            subdivision_outer_vertices = [
                outer_vertices_top[subdivision_index],
                outer_vertices_bottom[subdivision_index],
                outer_vertices_bottom[subdivision_index+1],
                outer_vertices_top[subdivision_index+1]
            ]
            subdivision_outer_vertices = rotate_list(current_list=subdivision_outer_vertices, steps=-1)

            subdivision_face.outer_vertices_vec3 = subdivision_outer_vertices
            subdivision_face.outer_vertices = vec3_list_to_list(subdivision_outer_vertices)
            subdivision_face.vertices = subdivision_face.outer_vertices_to_vertices()
            subdivision_face.normal = face.normal
            subdivision_faces.append(subdivision_face)
            subdivision_index += 1

        updated_faces += subdivision_faces
        return updated_faces

class PrimativeMeshEmission(PrimativeMesh):
    """
    Version of primative mesh that redefines draw() method to allow for
    emission texture.
    """
    def __init__(
        self,
        shader,
        # material properties
        diffuse,
        specular,
        emission,
        shininess=32.0,
        # mesh properties
        dimensions=[5.0, 5.0],
        position=[0.0, 0.0, 0.0],
        rotation_magnitude=(0.0, 0.0, 0.0),
        rotation_axis=(0.0, 0.0, 1.0),
        scale=(1.0, 1.0, 1.0),
    ):
        super().__init__(
            shader=shader,
            # material properties
            diffuse=diffuse,
            specular=specular,
            shininess=shininess,
            # mesh properties
            dimensions=dimensions,
            position=position,
            rotation_magnitude=glm.vec3(rotation_magnitude),
            rotation_axis=glm.vec3(rotation_axis),
            scale=glm.vec3(scale),
        )
        self.emission = emission
    def draw(self, view):
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.diffuse)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.specular)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, self.emission)

        # rotate, translate, and scale
        model = glm.mat4(1.0)
        model = glm.translate(model, glm.vec3(self.position))
        model = glm.rotate(model, self.rotation_magnitude.x, glm.vec3([1.0, 0.0, 0.0]))
        model = glm.rotate(model, self.rotation_magnitude.y, glm.vec3([0.0, 1.0, 0.0]))
        model = glm.rotate(model, self.rotation_magnitude.z, glm.vec3([0.0, 0.0, 1.0]))

        model = glm.scale(model, self.scale)
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, glm.value_ptr(model))
        glDrawArrays(GL_TRIANGLES, 0, int(len(self.vertices) / 3))

    def set_diffuse(self, diffuse):
        self.diffuse = diffuse
    def set_specular(self, specular):
        self.specular = specular
    def set_emission(self, emission):
        self.emission = emission

class Spaceship(PrimativeMeshEmission):
    cardinal_directions = {
        'down': [1.0, 0.0, 0.0],
        'up': [1.0, 0.0, 0.0],
        'port': [0.0, 0.0, 1.0],
        'starboard': [0.0, 0.0, -1.0],
        'forward': [0.0, 1.0, 0.0],
        'backward': [0.0, -1.0, 0.0],
    }

    def __init__(
        self,
        shader,
        # material properties
        diffuse,
        specular,
        emission,
        shininess=32.0,
        # mesh properties
        dimensions=[5.0, 5.0],
        position=[0.0, 0.0, 0.0],
        rotation_magnitude=0,
        rotation_axis=glm.vec3([0.0, 0.0, 1.0]),
        scale=glm.vec3([1.0, 1.0, 1.0]),
        number_of_sides=4,
        number_of_segments=4,
        transform_x=1.0,
        transform_z=1.0,
        length_of_segment=1.0,
        radius=1.0,
        seed=1,
    ):
        self.number_of_sides = number_of_sides
        self.number_of_segments = number_of_segments
        self.transform_x = transform_x
        self.transform_z = transform_z
        self.length_of_segment = length_of_segment
        self.radius = radius
        r.seed(seed)
        super().__init__(
            shader=shader,
            # material properties
            diffuse=diffuse,
            specular=specular,
            shininess=shininess,
            emission=emission,
            # mesh properties
            dimensions=dimensions,
            position=position,
            rotation_magnitude=rotation_magnitude,
            rotation_axis=rotation_axis,
            scale=scale,
        )
    def generate_vertices(self):
        """
        Where the mesh is generated, vertex by vertex.
        :return:
        """
        faces = [
            self._generate_polygonal_face(
                number_of_sides=self.number_of_sides,
                radius=self.radius,
                transform_x=self.transform_x,
                transform_z=self.transform_z,
            )
        ]
        faces[0].update_texture_coords_using_atlas_index(texture_atlas_index=1, texture_atlas_size=2)

        # generate the latter faces via extruding
        segment_faces = []
        radius_multiplier_current_segment = 1.0
        current_radius = self.radius
        MINIMUM_RADIUS = 2.5
        MAXIMUM_RADIUS = 4.0 + self.number_of_segments*0.5
        for i in range(self.number_of_segments):
            face_extruded = Face()
            radius_multiplier_current_segment = self.determine_radius(
                i,
                self.number_of_segments,
                radius_multiplier_current_segment,
                minimum_radius_multipler=0.5,
                maximum_radius_multipler=2.0,
                growth=1.0 + random()*3,
                shrinkage=random()*0.5 + 0.5,

            )
            current_radius *= radius_multiplier_current_segment
            if current_radius < MINIMUM_RADIUS or current_radius > MAXIMUM_RADIUS:
                radius_multiplier_current_segment = 1.0
            # old extrude-want to replace with one that does flip the face around properly for texturing
            face_extruded.extrude_from_other_face(
                other=faces[i],
                direction=list(faces[i].calculate_normal()),
                distance=self.length_of_segment,
                flip_base=True,
                scale=radius_multiplier_current_segment
            )
            faces.append(face_extruded)
            faces_from_stitch = self.stitch_faces(
                face_a=faces[i],
                face_b=faces[i+1],
                number_of_faces=self.number_of_sides
            )
            segment_faces += faces_from_stitch
        faces[-1].update_texture_coords_using_atlas_index(
            texture_atlas_index=3,
            texture_atlas_size=2,
        )
        faces += self.generate_thrusters(face=faces[-1])
        faces += self.add_detail_to_faces(faces_to_alter=segment_faces)
        faces += self.generate_nose(face=faces[0])
        del faces[0]

        return self.serialize_faces(faces)

    def generate_thrusters(self, face):
        thruster_faces = self.bevel_cut(
            original_face=face,
            bevel_depths=[self.length_of_segment*0.25, 0.0, self.length_of_segment*-0.5],
            border_sizes=[-0.25, 0.25, 0.5],
            depth=3
        )

        #texturing
        thruster_backend = thruster_faces.pop(-(self.number_of_sides + 1))
        thruster_backend.update_texture_coords_using_atlas_index(texture_atlas_size=2, texture_atlas_index=0)
        for face in thruster_faces:
            face.update_texture_coords_using_atlas_index(texture_atlas_size=2, texture_atlas_index=3)

        return thruster_faces + [thruster_backend]
    def generate_nose(self, face):
        faces_nose = []
        face_nose_tip = Face()
        face_nose_tip.extrude_from_other_face(
            other=face,
            direction=list(-face.calculate_normal()),
            distance=self.length_of_segment * (0.25 + random()*1.25),
            scale=random()*.2 + .1,
        )

        faces_from_stitch = self.stitch_faces(
            face_a=face_nose_tip,
            face_b=face,
            number_of_faces=self.number_of_sides
        )

        face_nose_tip.calculate_outer_vertices_as_vec3()
        face_nose_tip.outer_vertices_vec3.reverse()
        face_nose_tip.outer_vertices = vec3_list_to_list(face_nose_tip.outer_vertices_vec3)
        face_nose_tip.vertices = face_nose_tip.outer_vertices_to_vertices(
            reverse_texture_coordinates=True,
        )

        #texturing
        #todo:reenable
        face_nose_tip.update_texture_coords_using_atlas_index(texture_atlas_size=2, texture_atlas_index=1)
        for face in faces_from_stitch:
            face.update_texture_coords_using_atlas_index(texture_atlas_size=2, texture_atlas_index=2)

        #return new faces
        faces_nose += [face_nose_tip] + faces_from_stitch
        return faces_nose

    def determine_radius(
            self,
            i,
            number_of_segments,
            radius_multiplier,
            minimum_radius_multipler,
            maximum_radius_multipler,
            growth=1.25,
            shrinkage=0.5,
    ):

        if i < floor(random() * number_of_segments):
            radius_multiplier = min(growth*radius_multiplier, maximum_radius_multipler)
        else:
            if radius_multiplier > 1.0:
                radius_multiplier = 1.0
            radius_multiplier = max(shrinkage*radius_multiplier, minimum_radius_multipler)
        return radius_multiplier

    # def add_detail_to_faces(self, faces_to_alter):
    #     faces_altered = []
    #     for face in faces_to_alter:
    #         if random() < 0.25:
    #             current_stitch_faces = []
    #             current_stitch_faces += self.subdivide_quad_lengthwise(face=face, subdivisions=2, widthwise=True)
    #             for new_stitch_face in current_stitch_faces:
    #                 faces_altered += self.bevel_cut(
    #                     original_face=new_stitch_face,
    #                     bevel_depths=[-0.33],
    #                     border_sizes=[0.6],
    #                     depth=1,
    #                 )
    #         elif random() < 0.25:
    #             current_stitch_faces = []
    #             current_stitch_faces += self.pyrimidize_face(original_face=face, center_point_offset=2.5)
    #             faces_altered += current_stitch_faces
    #         elif random() < 0.5:
    #             current_stitch_faces = self.extrude_and_stitch(face, scale=1.0, distance=floor(self.length_of_segment*0.25))
    #             recursion_count = 1
    #             for i in range(recursion_count):
    #                 final_faces = []
    #                 for face in current_stitch_faces:
    #                     final_faces += self.extrude_and_stitch(face, scale=1.0, distance=floor(self.length_of_segment*0.25))
    #                 current_stitch_faces = final_faces
    #
    #             faces_post_bevel = []
    #             for face in final_faces:
    #                 faces_post_bevel += self.bevel_cut(
    #                     original_face=face,
    #                     bevel_depths=[0.33],
    #                     border_sizes=[0.6],
    #                     depth=1,
    #                 )
    #             faces_altered += faces_post_bevel
    #
    #         else:
    #             faces_altered += [face]
    #     for face in faces_altered:
    #         face.update_texture_coords_using_atlas_index(texture_atlas_index=2, texture_atlas_size=2)
    #     return faces_altered

    def add_detail_to_faces(self, faces_to_alter):
        """
        Goes around faces of segmented cylindar and replaces plain quads with detailed meshes
        :param faces_to_alter:
        :return:
        """

        if self.number_of_sides % 3 == 0:
            symmetry_type = 'triangular'
        elif self.number_of_sides % 2 == 0:
            symmetry_type = 'square'
        else:
            symmetry_type = 'irregular'

        instructions_per_segment = []
        for index_segment in range(self.number_of_segments):
            if symmetry_type == 'square':
                number_of_pairs = int(self.number_of_sides / 2)
                instructions_per_pair = []
                for i in range(number_of_pairs):
                    instructions_per_pair.append(random())
                instructions_per_segment.append(instructions_per_pair)
            elif symmetry_type == 'triangular':
                # instructions_per_segment.append((random(), random(), random()))
                number_of_triplets = int(self.number_of_sides / 3)
                instructions_per_pair = []
                for i in range(number_of_triplets):
                    instructions_per_pair.append(random())
                instructions_per_segment.append(instructions_per_pair)
            else:
                instructions_per_segment.append((random(),) * self.number_of_sides)

        faces_altered = []
        faces_unaltered = []
        for index_segment in range(self.number_of_segments):

            faces_paired = {}
            if symmetry_type == 'square':
                number_of_pairs = int(self.number_of_sides / 2)
                for index_pair in range(number_of_pairs):
                    if self.number_of_sides == 8:
                        if index_pair == 1 or index_pair == 3:
                            faces_paired[index_pair] = [
                                faces_to_alter[index_segment * self.number_of_sides + index_pair],
                                faces_to_alter[index_segment * self.number_of_sides + index_pair + number_of_pairs]
                            ]
                        elif index_pair == 0:
                            faces_paired[index_pair] = [
                                faces_to_alter[index_segment * self.number_of_sides + index_pair],
                                faces_to_alter[index_segment * self.number_of_sides + 6]
                            ]
                        else:
                            faces_paired[index_pair] = [
                                faces_to_alter[index_segment * self.number_of_sides + 4],
                                faces_to_alter[index_segment * self.number_of_sides + 2]
                            ]
                    else:
                        faces_paired[index_pair] = [
                            faces_to_alter[index_segment * self.number_of_sides + index_pair],
                            faces_to_alter[index_segment * self.number_of_sides + index_pair + number_of_pairs]
                        ]
                    faces_altered_local, faces_unaltered_local = self.detail_faces_by_instruction(
                        faces_altered=faces_altered,
                        index_segment=index_segment,
                        instructions_per_segment=instructions_per_segment,
                        faces_paired=faces_paired[index_pair],
                        index_instruction=index_pair,
                    )
                    faces_altered += faces_altered_local
                    faces_unaltered += faces_unaltered_local
            if symmetry_type == 'triangular':
                number_of_triplets = int(self.number_of_sides / 3)
                for index_pair in range(number_of_triplets):
                    faces_paired[index_pair] = [
                        faces_to_alter[index_segment * self.number_of_sides + index_pair],
                        faces_to_alter[index_segment * self.number_of_sides + index_pair + number_of_triplets],
                        faces_to_alter[index_segment * self.number_of_sides + index_pair + number_of_triplets * 2],
                    ]
                    faces_altered_local, faces_unaltered_local = self.detail_faces_by_instruction(
                        faces_altered=faces_altered,
                        index_segment=index_segment,
                        instructions_per_segment=instructions_per_segment,
                        faces_paired=faces_paired[index_pair],
                        index_instruction=index_pair,
                    )
                    faces_altered += faces_altered_local
                    faces_unaltered += faces_unaltered_local
            if symmetry_type == 'irregular':
                for index_side in range(self.number_of_sides):
                    faces_altered_local, faces_unaltered_local = self.detail_faces_by_instruction(
                        faces_altered=faces_altered,
                        index_segment=index_segment,
                        instructions_per_segment=instructions_per_segment,
                        faces_paired=[faces_to_alter[index_side + index_segment*self.number_of_sides]],
                        index_instruction=index_side,
                    )
                    faces_altered += faces_altered_local
                    faces_unaltered += faces_unaltered_local


        #todo: texturing should happen within detail-applying-functions.
        # for face in faces_altered:
        #     face.update_texture_coords_using_atlas_index(texture_atlas_index=2, texture_atlas_size=2)
        for face in faces_unaltered:
            face.update_texture_coords_using_atlas_index(texture_atlas_index=0, texture_atlas_size=2)
        return faces_altered + faces_unaltered





        # faces_altered = []
        # for face in faces_to_alter:
        #     if random() < 0.25:
        #         faces_altered = self.add_detail_cubbies(face, faces_altered)
        #     elif random() < 0.25:
        #         faces_altered = self.add_detail_pyrimide(face, faces_altered)
        #     elif random() < 0.5:
        #         faces_altered = self.add_detail_recursive_extrude(face, faces_altered)
        #
        #     else:
        #         faces_altered += [face]
        # for face in faces_altered:
        #     face.update_texture_coords_using_atlas_index(texture_atlas_index=2, texture_atlas_size=2)
        # return faces_altered

    def add_detail_recursive_extrude(self, face, bevel_depths, border_sizes):

        first_extrusion = self.extrude_and_stitch(face=face, scale=1.0, distance=floor(self.length_of_segment*0.25))
        faces_updated = []
        for face_current in first_extrusion:
            faces_one_bevel = self.bevel_cut(
                original_face=face_current,
                bevel_depths=bevel_depths[1:2],
                border_sizes=border_sizes[1:2],
                depth=1
            )
            faces_one_bevel[0].update_texture_coords_using_atlas_index(texture_atlas_index=1, texture_atlas_size=2)
            for face in faces_one_bevel:
                face.update_texture_coords_using_atlas_index(texture_atlas_index=2, texture_atlas_size=2)
            faces_updated += faces_one_bevel
        return faces_updated

    def add_detail_intake(self, face):
        local_faces_altered = self.bevel_cut(
            original_face=face,
            depth=1,
            bevel_depths=[self.length_of_segment*0.2],
            border_sizes=[0.0]
        )
        intake_index = 2
        intake_faces = [local_faces_altered[intake_index]]
        intake_faces_altered = []
        for face in intake_faces:
            intake_faces_altered += self.bevel_cut(
                original_face=face,
                depth=2,
                bevel_depths=[0.0, -self.length_of_segment * 0.3],
                border_sizes=[0.15, 0.5]
            )
        del local_faces_altered[intake_index]
        for face in local_faces_altered:
            face.update_texture_coords_using_atlas_index(texture_atlas_index=2, texture_atlas_size=2)
        for face in intake_faces_altered:
            face.update_texture_coords_using_atlas_index(texture_atlas_index=0, texture_atlas_size=2)
        intake_faces_altered[-5].update_texture_coords_using_atlas_index(texture_atlas_index=3, texture_atlas_size=2)

        # del local_faces_altered[4]

        return local_faces_altered + intake_faces_altered

    def detail_faces_by_instruction(
        self,
        faces_altered,
        index_segment,
        instructions_per_segment,
        faces_paired,
        index_instruction,
    ):
        local_faces_altered = []
        local_faces_unaltered = []

        for face in faces_paired:
            if instructions_per_segment[index_segment][index_instruction] < 0.25:
                # pass
                local_faces_altered += self.add_detail_cubbies(face)
            elif instructions_per_segment[index_segment][index_instruction] < 0.5:
                local_faces_altered += self.add_detail_pyrimide(face)
            elif instructions_per_segment[index_segment][index_instruction] < 0.73:
                local_faces_altered += self.add_detail_recursive_extrude(
                    face,
                    bevel_depths=[0.4, 0.33],
                    border_sizes=[0.0, 0.5]
                )
                for face in local_faces_altered:
                    if math.isnan(face.outer_vertices[0]):
                        print('outer vertex is nan!')
            elif instructions_per_segment[index_segment][index_instruction] < 0.75:
                local_faces_altered += self.add_detail_intake(
                    face,
                )

            elif instructions_per_segment[index_segment][index_instruction] < 0.9:
                faces_special_bevel = self.bevel_cut(
                    original_face=face,
                    bevel_depths=[0.0, -0.5],
                    border_sizes=[0.25, 0.0],
                    depth=2,
                )
                faces_special_bevel[-5].update_texture_coords_using_atlas_index(texture_atlas_index=3, texture_atlas_size=2)
                for face in faces_special_bevel[:-5]:
                    face.update_texture_coords_using_atlas_index(texture_atlas_index=2, texture_atlas_size=2)
                for face in faces_special_bevel[-4:]:
                    face.update_texture_coords_using_atlas_index(texture_atlas_index=2, texture_atlas_size=2)
                local_faces_altered += faces_special_bevel
            else:
                local_faces_unaltered += [face]
        return local_faces_altered, local_faces_unaltered

    def add_detail_pyrimide(self, face):
        current_stitch_faces = []
        faces_altered_local = []
        current_stitch_faces += self.pyrimidize_face(original_face=face, center_point_offset=2.5)
        faces_altered_local += current_stitch_faces
        for face in faces_altered_local:
            face.update_texture_coords_using_atlas_index(
                texture_atlas_index=2,
                texture_atlas_size=2
            )
        return faces_altered_local

    def add_detail_cubbies(self, face):
        current_stitch_faces = []
        current_stitch_faces += self.subdivide_quad_lengthwise(face=face, subdivisions=2, widthwise=True)
        faces_altered_local = []
        for new_stitch_face in current_stitch_faces:
            faces_current_cubby = self.bevel_cut(
                original_face=new_stitch_face,
                bevel_depths=[-0.33],
                border_sizes=[0.5],
                depth=1,
            )
            faces_current_cubby[0].update_texture_coords_using_atlas_index(
                texture_atlas_index=3, #todo: change to 1
                texture_atlas_size=2
            )
            for face in faces_current_cubby[1:]:
                face.update_texture_coords_using_atlas_index(
                    texture_atlas_index=2, #todo: change to 2
                    texture_atlas_size=2
                )
            faces_altered_local += faces_current_cubby
        return faces_altered_local

    def extrude_and_stitch(self, face, number_of_faces=4, distance=1, scale=1.0):
        current_stitch_faces = []
        face.calculate_outer_vertices_as_vec3()
        face_extrude = Face()
        face_normal = face.calculate_normal()
        face_extrude.extrude_from_other_face(other=face, direction=face_normal, distance=distance, scale=scale)
        current_stitch_faces += [face_extrude]
        current_stitch_faces += self.stitch_faces(face_a=face, face_b=face_extrude, number_of_faces=number_of_faces)
        return current_stitch_faces

    def serialize_faces(self, faces):
        """
        Convert vertices of our model from our abstract Faces format to serialized data (a Numpy array) that
        OpenGL can draw.
        :param faces:
        :return:
        """

        vertices = []
        for face in faces:
            vertices += face.face_to_list()
        faces = np.array(
            vertices,
            dtype=np.float32
        ).flatten()
        return faces


class Polygon(PrimativeMesh):
    """
    Generates an N-sided Polygonal face
    """

    def __init__(self,
        shader,
        diffuse,
        specular,
        shininess=32.0,
        dimensions=[5.0, 5.0],
        position=[0.0, 0.0, 0.0],
        rotation_magnitude=0,
        rotation_axis=glm.vec3([0.0, 0.0, 1.0]),
        scale=glm.vec3([1.0, 1.0, 1.0]),
        sides=3,
        transform_x=1.0,
        transform_z=1.0,
     ):
        self.sides = sides
        self.transform_x = transform_x
        self.transform_z = transform_z
        super().__init__(
            shader=shader,
            diffuse=diffuse,
            specular=specular,
            shininess=shininess,
            dimensions=dimensions,
            position=position,
            rotation_magnitude=rotation_magnitude,
            rotation_axis=rotation_axis,
            scale=scale,
        )

    def generate_vertices(self):
        """
        Generates the outer vertices and vertices of the triangle of this model
        """
        face = self._generate_polygonal_face(
            number_of_sides=self.sides,
            transform_x=self.transform_x,
            transform_z=self.transform_z,
        )
        vertices = face.face_to_list()
        return np.array(
            vertices,
            dtype=np.float32
        ).flatten()

class PolygonIrregular(Polygon):
    def __init__(
            self,
            shader,
            diffuse,
            specular,
            shininess=32.0,
            dimensions=[5.0, 5.0],
            position=[0.0, 0.0, 0.0],
            rotation_magnitude=0,
            rotation_axis=glm.vec3([0.0, 0.0, 1.0]),
            scale=glm.vec3([1.0, 1.0, 1.0]),
            sides=3,
            transform_x=1.0,
            transform_z=1.0,
            radii=1.0,
    ):
        self.radii = radii
        super().__init__(
            shader=shader,
            diffuse=diffuse,
            specular=specular,
            shininess=shininess,
            dimensions=dimensions,
            position=position,
            rotation_magnitude=rotation_magnitude,
            rotation_axis=rotation_axis,
            scale=scale,
            sides=sides,
            transform_x=transform_x,
            transform_z=transform_z,
        )

    def generate_vertices(self):
        """
        Generates the outer vertices and vertices of the triangle of this model
        """
        face = self._generate_irregular_polygonal_face(
            number_of_sides=self.sides,
            transform_x=self.transform_x,
            transform_z=self.transform_z,
            radii=self.radii,
        )
        vertices = face.face_to_list()
        return np.array(
            vertices,
            dtype=np.float32
        ).flatten()
class QuadMesh(PrimativeMesh):
    pass

class TriangleMesh(PrimativeMesh):
    """
    simple equilateral triangle.  oriented with normal in the Z+ direction, lying upon XY Plane, at origin
    """

    def generate_vertices(self):
        top_vertex_angle = math.pi / 2.0
        return np.array([
             self.dimensions[0]*math.cos(top_vertex_angle), self.dimensions[0]*math.sin(top_vertex_angle), 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
             self.dimensions[0]*math.cos(top_vertex_angle + 2.0*math.pi/3.0),  self.dimensions[0]*math.sin(top_vertex_angle + 2.0*math.pi/3.0), 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
             self.dimensions[0]*math.cos(top_vertex_angle + 4.0*math.pi/3.0),  self.dimensions[0]*math.sin(top_vertex_angle + 4.0*math.pi/3.0), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            dtype=np.float32
        )

class CubeMeshStatic(PrimativeMesh):
    """
    Gives us a cube mesh, but does not proc-gen the cube using our face-and-excude system
    my_cube = primatives.CubeMesh(
    shader=shader,
    diffuse=tile_textures[0],
    specular=tile_textures[1],
    shininess=32.0,
    position=[0.0, 0.0, 10.0],
    dimensions=[1.0, 3.0, 5.0],
    rotation_axis=[0.5, 1.0, 0.1]
    )

    How to draw:
    view = active_cam.get_view_matrix()
    my_cube.draw(view=view)
    """

    def generate_vertices(self):
        return np.array([
              #Z- facing
             -1.0*self.dimensions[0], -1.0*self.dimensions[1], -1.0*self.dimensions[2], 0.0, 0.0,  0.0, 0.0, -1.0,
              1.0*self.dimensions[0],  1.0*self.dimensions[1], -1.0*self.dimensions[2], 1.0, 1.0,  0.0, 0.0, -1.0,
              1.0*self.dimensions[0], -1.0*self.dimensions[1], -1.0*self.dimensions[2], 1.0, 0.0,  0.0, 0.0, -1.0,
              1.0*self.dimensions[0],  1.0*self.dimensions[1], -1.0*self.dimensions[2], 1.0, 1.0,  0.0, 0.0, -1.0,
             -1.0*self.dimensions[0], -1.0*self.dimensions[1], -1.0*self.dimensions[2], 0.0, 0.0,  0.0, 0.0, -1.0,
             -1.0*self.dimensions[0],  1.0*self.dimensions[1], -1.0*self.dimensions[2], 0.0, 1.0,  0.0, 0.0, -1.0,
            #Z+ facing
            -1.0*self.dimensions[0], -1.0*self.dimensions[1], 1.0*self.dimensions[2], 0.0, 0.0, 0.0, 0.0,   1.0,
             1.0*self.dimensions[0], -1.0*self.dimensions[1], 1.0*self.dimensions[2], 1.0, 0.0,  0.0, 0.0,  1.0,
             1.0*self.dimensions[0],  1.0*self.dimensions[1], 1.0*self.dimensions[2], 1.0, 1.0,   0.0, 0.0, 1.0,
             1.0*self.dimensions[0],  1.0*self.dimensions[1], 1.0*self.dimensions[2], 1.0, 1.0,   0.0, 0.0, 1.0,
            -1.0*self.dimensions[0],  1.0*self.dimensions[1], 1.0*self.dimensions[2], 0.0, 1.0,  0.0, 0.0,  1.0,
            -1.0*self.dimensions[0], -1.0*self.dimensions[1], 1.0*self.dimensions[2], 0.0, 0.0, 0.0, 0.0,   1.0,
            #X- facing
            -1.0*self.dimensions[0],  1.0*self.dimensions[1],  1.0*self.dimensions[2], 1.0, 1.0, -1.0, 0.0, 0.0,
            -1.0*self.dimensions[0],  1.0*self.dimensions[1], -1.0*self.dimensions[2], 0.0, 1.0, -1.0, 0.0, 0.0,
            -1.0*self.dimensions[0], -1.0*self.dimensions[1], -1.0*self.dimensions[2], 0.0, 0.0, -1.0, 0.0, 0.0,
            -1.0*self.dimensions[0], -1.0*self.dimensions[1], -1.0*self.dimensions[2], 0.0, 0.0, -1.0, 0.0, 0.0,
            -1.0*self.dimensions[0], -1.0*self.dimensions[1],  1.0*self.dimensions[2], 1.0, 0.0, -1.0, 0.0, 0.0,
            -1.0*self.dimensions[0],  1.0*self.dimensions[1],  1.0*self.dimensions[2], 1.0, 1.0, -1.0, 0.0, 0.0,
            #X+ facing
            1.0*self.dimensions[0],  1.0*self.dimensions[1],  1.0*self.dimensions[2], 1.0, 1.0, 1.0, 0.0, 1.0,
            1.0*self.dimensions[0], -1.0*self.dimensions[1], -1.0*self.dimensions[2], 0.0, 0.0, 1.0, 0.0, 1.0,
            1.0*self.dimensions[0],  1.0*self.dimensions[1], -1.0*self.dimensions[2], 0.0, 1.0, 1.0, 0.0, 1.0,
            1.0*self.dimensions[0], -1.0*self.dimensions[1], -1.0*self.dimensions[2], 0.0, 0.0, 1.0, 0.0, 1.0,
            1.0*self.dimensions[0],  1.0*self.dimensions[1],  1.0*self.dimensions[2], 1.0, 1.0, 1.0, 0.0, 1.0,
            1.0*self.dimensions[0], -1.0*self.dimensions[1],  1.0*self.dimensions[2], 1.0, 0.0, 1.0, 0.0, 1.0,
            #Y- facing
            -1.0*self.dimensions[0], -1.0*self.dimensions[1], -1.0*self.dimensions[2], 0.0, 1.0,  0.0, -1.0, 0.0,
             1.0*self.dimensions[0], -1.0*self.dimensions[1], -1.0*self.dimensions[2], 1.0, 1.0,  0.0, -1.0, 0.0,
             1.0*self.dimensions[0], -1.0*self.dimensions[1],  1.0*self.dimensions[2], 1.0, 0.0,  0.0, -1.0, 0.0,
             1.0*self.dimensions[0], -1.0*self.dimensions[1],  1.0*self.dimensions[2], 1.0, 0.0,  0.0, -1.0, 0.0,
            -1.0*self.dimensions[0], -1.0*self.dimensions[1],  1.0*self.dimensions[2], 0.0, 0.0,  0.0, -1.0, 0.0,
            -1.0*self.dimensions[0], -1.0*self.dimensions[1], -1.0*self.dimensions[2], 0.0, 1.0,  0.0, -1.0, 0.0,
            #Y+ facing
            -1.0*self.dimensions[0], 1.0*self.dimensions[1], -1.0*self.dimensions[2], 0.0, 1.0, 0.0, 1.0, 0.0,
             1.0*self.dimensions[0], 1.0*self.dimensions[1],  1.0*self.dimensions[2], 1.0, 0.0, 0.0, 1.0, 0.0,
             1.0*self.dimensions[0], 1.0*self.dimensions[1], -1.0*self.dimensions[2], 1.0, 1.0, 0.0, 1.0, 0.0,
             1.0*self.dimensions[0], 1.0*self.dimensions[1],  1.0*self.dimensions[2], 1.0, 0.0, 0.0, 1.0, 0.0,
            -1.0*self.dimensions[0], 1.0*self.dimensions[1], -1.0*self.dimensions[2], 0.0, 1.0, 0.0, 1.0, 0.0,
            -1.0*self.dimensions[0], 1.0*self.dimensions[1],  1.0*self.dimensions[2], 0.0, 0.0, 0.0, 1.0, 0.0,],
            dtype=np.float32
        )

class FloorTileMesh(PrimativeMesh):
    """
    generates a basic floor-tile for our little proc-gen roguelike
    """

    def __init__(self,
                 shader,
                 # material properties
                 diffuse,
                 specular,
                 shininess=32.0,
                 # mesh properties
                 dimensions=[5.0, 5.0],
                 position=[0.0, 0.0, 0.0],
                 rotation_magnitude=0,
                 rotation_axis=glm.vec3([0.0, 0.0, 1.0]),
                 scale=glm.vec3([1.0, 1.0, 1.0]),
                 ):
        self.shader = shader
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.position = position
        self.rotation_magnitude = glm.vec3(rotation_magnitude)
        self.rotation_axis = glm.vec3(rotation_axis)
        self.scale = glm.vec3(scale)
        self.dimensions = dimensions
        self.texture_horizontal_offset = [0.0, 0.0]
        self.texture_vertical_offset = [0.0, 0.0]
        self.vertices = self.generate_vertices()


        # quad VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        # quad position vertices (vertex attribute)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # quad texture coords
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 8, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        # quad normals
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 8, ctypes.c_void_p(20))
        glEnableVertexAttribArray(2)

        self.model_loc = glGetUniformLocation(self.shader, "model")

    def generate_vertices(self):
        #todo: seperate position from texture coord and normal
            #then zip-up at then end
        #todo: excude base
        #todo: stitch two bases together (Get a prism)
        #todo: make new tile class, based on this one, that will
            #1. outline points for a new base ontop of the floor
            #2. excude upwards
            #3. stitch together to get a wall
        #todo: organize by surface
            #a new class that holds vertices of a flat, polygonal surface
            #cut a hole in the surface, re-stitch the triangles
            #excude from a hole
        #bezier curved-holes should be final touch
            #focus on cutting, excuding, bevel(cut + excude + fill)
        radius = self.dimensions[0]
        number_of_sides = 4.0
        initial_angle = pi/number_of_sides

        angle_of_rotation = 2*pi/number_of_sides
        vertices = []
        faces = []
        faces.append(Face())

        faces[0].outer_vertices = self.generate_outer_vertices(number_of_sides=number_of_sides, initial_angle=initial_angle)
        faces[0].offset_outer_vertices(offset=glm.vec3([0.0, -1.0, 0.0]))
        faces[0].vertices = self.outer_vertices_to_vertices(number_of_sides=number_of_sides, face=faces[0])
        faces[0].apply_hardset_quad_texture_coords()

        #generate the second face via exude
        faces.append(Face())
        faces[1].extrude_from_other_face(other=faces[0], direction = [0.0, 1.0, 0.0], distance=self.dimensions[1])

        #generate the 4 vertical faces
        faces += self.stitch_faces(face_a=faces[0], face_b=faces[1], number_of_faces=number_of_sides)


        vertices = []
        for face in faces:
            vertices += face.face_to_list()
        return np.array(vertices,
            dtype=np.float32
        ).flatten()

    def generate_outer_vertices(self, number_of_sides=4, initial_angle=None):
        """
        generates the vertices of a simple polygon-no repeats
        used for excuding and stitching, not for drawing
        """
        outer_vertices = []
        angle_of_rotation = 2 * pi / number_of_sides
        if initial_angle == None:
            initial_angle = pi / number_of_sides
        for i in range(1, int(number_of_sides) + 1):
            x_coord = cos(initial_angle + angle_of_rotation * -i) * self.dimensions[0]
            z_coord = sin(initial_angle + angle_of_rotation * -i) * self.dimensions[0]
            outer_vertices += [x_coord, 0.0, z_coord]
        return outer_vertices

    def outer_vertices_to_vertices(self, face, number_of_sides):
        """
        Generates the real vertices (with normals, texture coords) of a face from the
        outer vertices
        :param face: the face we want to generate vertices for.  must have its outer vertices already
        :param number_of_sides: eg, 4 for square, 5 for pentagon
        :return: the vertices to be drawn by opengl
        """

        vertices = []
        for i in range(int(number_of_sides) - 2):
            triangle_vertices = [Vertex(), Vertex(), Vertex()]

            triangle_vertices[0].positions = glm.vec3([
                 face.outer_vertices[0],
                 face.outer_vertices[1],
                 face.outer_vertices[2]
            ])
            triangle_vertices[0].texture_coordinates = glm.vec2([
                face.outer_vertices[0] / 2.0 + 0.5,
                face.outer_vertices[2] / 2.0 + 0.5
            ])
            triangle_vertices[0].normals = glm.vec3([0.0, 1.0, 0.0])


            triangle_vertices[1].positions = glm.vec3([
                face.outer_vertices[3 * i + 0 + 3 * 1],
                face.outer_vertices[3 * i + 1 + 3 * 1],
                face.outer_vertices[3 * i + 2 + 3 * 1]
            ])
            triangle_vertices[1].texture_coordinates = glm.vec2([
                face.outer_vertices[3 * i + 0 + 3 * 1] / 2.0 + 0.5,
                face.outer_vertices[3 * i + 2 + 3 * 1] / 2.0 + 0.5
            ])
            triangle_vertices[1].normals = glm.vec3([0.0, 1.0, 0.0])

            triangle_vertices[2].positions = glm.vec3([
                face.outer_vertices[3 * i + 0 + 3 * 2],
                face.outer_vertices[3 * i + 1 + 3 * 2],
                face.outer_vertices[3 * i + 2 + 3 * 2]
            ])
            triangle_vertices[2].texture_coordinates = glm.vec2([
                face.outer_vertices[3 * i + 0 + 3 * 2] / 2.0 + 0.5,
                face.outer_vertices[3 * i + 2 + 3 * 2] / 2.0 + 0.5
            ])
            triangle_vertices[2].normals = glm.vec3([0.0, 1.0, 0.0])
            for vertex in triangle_vertices:
                vertex.normals = calculate_normal([
                    triangle_vertices[0].positions,
                    triangle_vertices[1].positions,
                    triangle_vertices[2].positions
                ])
            vertices += triangle_vertices
        return vertices

class Face():
    """
    convert Face into a 1-dimensional numpy array that opengl can draw
    """
    def __init__(self, vertices=[], outer_vertices=[]):
        self.vertices = vertices
        self.outer_vertices = outer_vertices #non-repeating vertices, to use for excuding, stitching, etc
        self.tile_width = 2.0 #TODO: make this a parameter
        self.texture_horizontal_offset = [0.0, 0.0]
        self.texture_vertical_offset = [0.0, 0.0]
        self.normal = glm.vec3([0.0, 1.0, 0.0])
        self.outer_vertices_vec3 = None
        self.radius = 1.0 #todo calculate this from outer_vertices or something
        self.sides = 0
        self.texture_atlas_index = 0

    def calculate_sides(self):
        self.calculate_outer_vertices_as_vec3()
        self.sides = len(self.outer_vertices_vec3)

    def offset_vertices_all(self, offset=glm.vec3([0.0, 0.0 ,0.0])):
        """
        offset the outer vertices and inner vertices

        :return: None
        """
        self.offset_outer_vertices(offset=offset)
        for vertex in self.vertices:
            vertex.positions += offset

    def calculate_normal(self, reverse=False):
        """
        uses outer vertices to calculate normal.
        :return: glm.vec3 representing the normal to this face's plane
        """
        try:
            triangle = [
                self.outer_vertices_vec3[0],
                self.outer_vertices_vec3[1],
                self.outer_vertices_vec3[2]
            ]
            if reverse:
                triangle = [
                    self.outer_vertices_vec3[0],
                    self.outer_vertices_vec3[2],
                    self.outer_vertices_vec3[1]
                ]
            normal = calculate_normal(
                triangle
            )
        except TypeError:
            self.calculate_outer_vertices_as_vec3()
            triangle = [
                self.outer_vertices_vec3[0],
                self.outer_vertices_vec3[1],
                self.outer_vertices_vec3[2]
            ]
            if reverse:
                triangle = [
                    self.outer_vertices_vec3[0],
                    self.outer_vertices_vec3[2],
                    self.outer_vertices_vec3[1]
                ]
            normal = calculate_normal(
                triangle
            )

        return normal

    def calculate_outer_vertices_as_vec3(self):
        """
        does not return a value, but assigns to attribute
        :return: nothing
        """
        self.outer_vertices_vec3 = list_to_vec3_list(self.outer_vertices)

    def calculate_radius(self):
        if self.outer_vertices_vec3 == None:
            self.calculate_outer_vertices_as_vec3()
        self.radius = glm.distance(calculate_mean_point(self.outer_vertices_vec3), self.outer_vertices_vec3[0])

    def set_texture_offsets(self, vertical, horizontal):
        self.texture_horizontal_offset = horizontal
        self.texture_vertical_offset = vertical

    def face_to_list(self):
        vertices_list = []
        for vertex in self.vertices:
            vertices_list.append(vertex.vertex_to_list())
        return vertices_list

    def extrude_from_other_face(
            self,
            other,
            direction=(0.0, 1.0, 0.0),
            distance=1.0,
            flip_base=True,
            scale=None,
    ):
        """
        takes in another face, copies points with positional offset.  Only effects outer vertices, so call to
        generate outer vertices is required as next step
        todo: this should live in primative mesh.  needs some refactoring to do so
        """
        offset = glm.vec3(direction) * distance
        all_vertices = []
        for vertex in other.vertices:
            updated_vertex = Vertex()
            if scale is None:
                updated_vertex.positions = glm.vec3(vertex.positions) + offset
            else:
                radius_adjusted_vertices = (
                    glm.vec3(vertex.positions).x*scale,
                    glm.vec3(vertex.positions).y,
                    glm.vec3(vertex.positions).z*scale
                )
                updated_vertex.positions = glm.vec3(radius_adjusted_vertices) + offset
            updated_vertex.texture_coordinates = glm.vec2(vertex.texture_coordinates)
            updated_vertex.normals = glm.vec3(vertex.normals)
            if flip_base:
                vertex.normals *= -1.0
            all_vertices.append(updated_vertex)
        if flip_base:
            other.vertices.reverse()
        self.vertices = all_vertices
        outer_vertices = [0.0] * len(other.outer_vertices)
        for vertex_index in range(int(len(other.outer_vertices)/3)):
            if scale is None:
                outer_vertices[vertex_index * 3] = other.outer_vertices[vertex_index * 3] + offset.x
                outer_vertices[vertex_index * 3 + 1] = other.outer_vertices[vertex_index * 3 + 1] + offset.y
                outer_vertices[vertex_index * 3 + 2] = other.outer_vertices[vertex_index * 3 + 2] + offset.z
            else:
                outer_vertices[vertex_index * 3] = other.outer_vertices[vertex_index * 3] * scale + offset.x
                outer_vertices[vertex_index * 3 + 1] = other.outer_vertices[vertex_index * 3 + 1] + offset.y
                outer_vertices[vertex_index * 3 + 2] = other.outer_vertices[vertex_index * 3 + 2] * scale + offset.z

        self.outer_vertices = outer_vertices.copy()

        self.calculate_sides()
        self.vertices = self.outer_vertices_to_vertices(reverse_texture_coordinates=True)

    def extrude_from_other_face_2(self, other, direction = [0.0, 1.0, 0.0], distance=1.0, flip_base=True, radius=1.0):
        """
        Trying a new extrude process WIP
        """
        offset = glm.vec3(direction) * distance
        all_vertices = []
        other.calculate_outer_vertices_as_vec3()
        outer_vertices_other = other.outer_vertices_vec3.copy()
        if flip_base:
            outer_vertices_other.reverse()
        self.outer_vertices_vec3 = outer_vertices_other.copy()
        for vertex in self.outer_vertices_vec3:
            vertex.x *= radius
            vertex.z *= radius
            vertex.x += offset.x
            vertex.y += offset.y
            vertex.z += offset.z
        if flip_base:
            self.outer_vertices = vec3_list_to_list(rotate_list(self.outer_vertices_vec3, math.ceil(len(self.outer_vertices_vec3)/2)))
        else:
            self.outer_vertices = vec3_list_to_list(self.outer_vertices_vec3)
        self.vertices = self.outer_vertices_to_vertices()

        #todo: this should work....
        for vertex, other_vertex in zip(self.vertices, other.vertices):
            vertex.texture_coordinates = glm.vec2(other_vertex.texture_coordinates)

    def update_texture_coords_using_atlas_index(self, texture_atlas_index, texture_atlas_size):
        """
        uses the texture atlas index to modify the per-vertex texture coordinates of this face

       :param texture_atlas_index: left->right, top->bottom
       :param texture_atlas_size: the length of the texture atlas (always square)
       :return:
       """
        self.texture_atlas_index = texture_atlas_index

        #texture atlas index into row and column indices
        column_index = float(texture_atlas_index % texture_atlas_size)
        row_index = 1.0 - math.floor(texture_atlas_index / texture_atlas_size)

        #row and column indices into lower and upper bounds of texture coords (4 total)
        #column (x-axis)lower and upper
        lower_texture_coord_x_axis = 1.0 / texture_atlas_size * column_index
        upper_texture_coord_x_axis = 1.0 / texture_atlas_size * (column_index + 1.0)
        #assuming square textures
        subtexture_magnitude = upper_texture_coord_x_axis - lower_texture_coord_x_axis
        lower_texture_coord_y_axis = 1.0 / texture_atlas_size * row_index
        upper_texture_coord_y_axis = 1.0 / texture_atlas_size * (row_index + 1.0)
        #update the current vertices' text coords
        for vertex in self.vertices:
            vertex.texture_coordinates[0] = lower_texture_coord_x_axis + vertex.texture_coordinates[0] * subtexture_magnitude
            vertex.texture_coordinates[1] = lower_texture_coord_y_axis + vertex.texture_coordinates[1] * subtexture_magnitude

    def apply_hardset_quad_texture_coords(self):
        """
        A brute-force fix to getting correct texture coords on the slab

        in future: a function that takes number of sides, and generates dictionary like below.
        outer_to_vertices function then reads this dictionary
            -found better solution in bezier_cut()
        """
        horizontal_offset = self.texture_horizontal_offset
        vertical_offset = self.texture_vertical_offset
        quad_texture_coords = {
            1: glm.vec2([0.0 + horizontal_offset[0], 0.0 + vertical_offset[0]]),
            0: glm.vec2([0.0 + horizontal_offset[0], 1.0 + vertical_offset[1]]),
            3: glm.vec2([1.0 + horizontal_offset[1], 1.0 + vertical_offset[1]]),
            2: glm.vec2([1.0 + horizontal_offset[1], 0.0 + vertical_offset[0]]),
        }

        self.vertices[0].texture_coordinates = glm.vec2(quad_texture_coords[0])
        self.vertices[1].texture_coordinates = glm.vec2(quad_texture_coords[1])
        self.vertices[2].texture_coordinates = glm.vec2(quad_texture_coords[2])
        self.vertices[3].texture_coordinates = glm.vec2(quad_texture_coords[0])
        self.vertices[4].texture_coordinates = glm.vec2(quad_texture_coords[2])
        self.vertices[5].texture_coordinates = glm.vec2(quad_texture_coords[3])

    def apply_hardset_triangle_texture_coords(self):
        """
        another hacky solution of calculating texture coordinates
        """
        self.vertices[0].texture_coordinates = glm.vec2([0.0, 0.0])
        self.vertices[1].texture_coordinates = glm.vec2([1.0, 0.0])
        self.vertices[2].texture_coordinates = glm.vec2([0.5, 1.0])

    def offset_outer_vertices(self, offset=glm.vec3([0.0, 0.0, 0.0])):
        for i in range(len(self.outer_vertices)):
            if i%3 == 0:
                self.outer_vertices[i] += offset.x
            if i%3 == 1:
                self.outer_vertices[i] += offset.y
            if i%3 == 2:
                self.outer_vertices[i] += offset.z
        # print(self.outer_vertices)

    def generate_texture_coordinates_regular_polygon(self, sides, reverse=True):
        """
        A new means of generating texture coordinates for any (regular) polygon.
        Even works for polygons not on XZ plane at origin
        :return: a list of glm.vec2() coordinates, one for each point of shape
        """
        angle_delta = -2.0 * pi / sides
        angle_initial = (pi / sides) - (pi / 2.0)
        radius = 0.5
        if sides == 4:
            radius = 1.0 / math.sqrt(2.0)
            angle_initial = (pi / sides) + angle_delta
        texture_coordinates = []
        for index_point in range(sides):
            coordinate = glm.vec2([0.0, 0.0])
            coordinate.x = (cos(angle_initial - angle_delta * index_point) * radius) + 0.5
            coordinate.y = (sin(angle_initial - angle_delta * index_point) * radius) + 0.5
            texture_coordinates.append(coordinate)
        if reverse:
            texture_coordinates.reverse()
        return texture_coordinates

    def generate_texture_coordinates_polygon(self, reverse=False):
        """
        Generate texture coordinates for any convex polygon (even irregular)
        in 3D space.
        :param reverse: reverse the texture coordinates at the end
        :return: list of coordinates as vec2
        """
        texture_coordinates = []
        self.calculate_outer_vertices_as_vec3()
        sides = len(self.outer_vertices_vec3)
        #1 get center point of polygon
        center_point = calculate_mean_point(points=self.outer_vertices_vec3)
        #get distances from center point to each vertex
            #also, find the largest distance
        vertex_distances = []
        largest_distance = 0
        for point in self.outer_vertices_vec3:
            distance = glm.distance(point, center_point)
            vertex_distances.append(distance)
            if distance > largest_distance:
                largest_distance = distance
        #scale the distances by the largest
        for index, distance in enumerate(vertex_distances):
            vertex_distances[index] = distance/(largest_distance*2)

        #Get the angles between the vertices
        vertex_angles = []
        angle_sum = 0
        for index in range(sides-1):
            a = self.outer_vertices_vec3[index]
            b = self.outer_vertices_vec3[index+1]
            c = center_point
            angle = self.calculate_angle(a, b, c)
            vertex_angles.append(angle)
            angle_sum += angle
        vertex_angles.append(2*pi - angle_sum)

        is_even = False
        if sides % 2 == 0:
            is_even = True
        #calculate initial angle offset (rotate the whole triangle)
        if is_even:
            if (sides/2) % 2 == 0:
                angle_initial = pi/2 + vertex_angles[0] + vertex_angles[1]/2
                if sides == 8:
                    angle_initial = pi + vertex_angles[2] + vertex_angles[1]/2
                elif sides == 12:
                    angle_initial = pi + vertex_angles[2]/2 + vertex_angles[0] + vertex_angles[1]
            else:
                #6
                angle_initial = pi + vertex_angles[0]
                if sides == 10:
                    angle_initial = pi + vertex_angles[0] + vertex_angles[1]
        else:
            angle_initial = pi + ((pi - vertex_angles[-1]) / 2)
        vertex_distances = rotate_list(vertex_distances, steps=1)

        #go around the unit circle, angle by angle, and place a point at the distance away
        for index in range(0, sides):
            angle_current = sum(vertex_angles[:index + 1])
            coordinate = glm.vec2((0.0, 0.0))
            coordinate.x = (cos(angle_initial - angle_current) * vertex_distances[index]) + 0.5
            coordinate.y = (sin(angle_initial - angle_current) * vertex_distances[index]) + 0.5
            texture_coordinates.append(coordinate)
        if is_even:
            texture_coordinates = rotate_list(texture_coordinates, steps=sides-1)
        else:
            texture_coordinates = rotate_list(texture_coordinates, steps=sides-1)
        if reverse:
            texture_coordinates.reverse()
        return texture_coordinates

    def calculate_angle(self, a, b, c):
        CA = a - c
        CB = b - c
        angle = glm.acos(glm.dot(CA, CB) / (glm.length(CA) * glm.length(CB)))
        return angle

    def outer_vertices_to_vertices(self, reverse_texture_coordinates=False, regular=False):
        """
        Generates the real vertices (with normals, texture coords) of a face
        :param regular: polygon is regular or irregular
        :param reverse_texture_coordinates: Flips texture coordinates about the textures Y-Axis
        :param number_of_sides: eg, 4 for square, 5 for pentagon
        :return: vertices ready to be drawn by opengl
        """
        if not self.sides:
            self.calculate_sides()
        number_of_sides = self.sides
        if not self.outer_vertices_vec3:
            self.calculate_outer_vertices_as_vec3()
        if regular:
            texture_coordinates = self.generate_texture_coordinates_regular_polygon(sides=number_of_sides)
        else:
            texture_coordinates = self.generate_texture_coordinates_polygon(reverse=reverse_texture_coordinates)

        vertices = []
        texture_index = 1
        for i in range(int(number_of_sides) - 2):
            triangle_vertices = [Vertex(), Vertex(), Vertex()]

            triangle_vertices[0].positions = glm.vec3((
                self.outer_vertices[0],
                self.outer_vertices[1],
                self.outer_vertices[2]
            ))
            triangle_vertices[0].texture_coordinates = glm.vec2((
                texture_coordinates[0].x,
                texture_coordinates[0].y
            ))
            triangle_vertices[0].normals = glm.vec3([0.0, 1.0, 0.0])

            triangle_vertices[1].positions = glm.vec3((
                self.outer_vertices[3 * i + 0 + 3 * 1],
                self.outer_vertices[3 * i + 1 + 3 * 1],
                self.outer_vertices[3 * i + 2 + 3 * 1]
            ))
            triangle_vertices[1].texture_coordinates = glm.vec2((
                texture_coordinates[texture_index].x,
                texture_coordinates[texture_index].y
            ))
            triangle_vertices[1].normals = glm.vec3([0.0, 1.0, 0.0])

            triangle_vertices[2].positions = glm.vec3((
                self.outer_vertices[3 * i + 0 + 3 * 2],
                self.outer_vertices[3 * i + 1 + 3 * 2],
                self.outer_vertices[3 * i + 2 + 3 * 2]
            ))
            triangle_vertices[2].texture_coordinates = glm.vec2((
                texture_coordinates[texture_index + 1].x,
                texture_coordinates[texture_index + 1].y
            ))
            triangle_vertices[2].normals = glm.vec3((0.0, 1.0, 0.0))
            for vertex in triangle_vertices:
                vertex.normals = calculate_normal([
                    triangle_vertices[0].positions,
                    triangle_vertices[1].positions,
                    triangle_vertices[2].positions
                ])
            vertices += triangle_vertices
            texture_index += 1
        return vertices

    def flip_winding_order(self):
        """
        flips this winding order of this face, mirroring vertices outer vertices
        """
        self.outer_vertices_vec3.reverse()
        self.outer_vertices = vec3_list_to_list(self.outer_vertices_vec3)
        self.outer_vertices_to_vertices()

class Vertex():
    def __init__(self):
        self.positions = glm.vec3()
        self.normals = glm.vec3()
        self.texture_coordinates = glm.vec2()

    def vertex_to_list(self):
        """
        convert positions, normals, and  texture coords into one big 1-dimensional list
        """
        vertex_as_list = []
        for position in self.positions:
            vertex_as_list.append(position)
        for texture_coord in self.texture_coordinates:
            vertex_as_list.append(texture_coord)
        for normal in self.normals:
            vertex_as_list.append(normal)
        return vertex_as_list

class WallTileMesh(FloorTileMesh):
    """
    straight walls for wfc world
    """
    def generate_vertices(self):
        radius = self.dimensions[0]
        number_of_sides = 4.0
        initial_angle = pi/number_of_sides

        angle_of_rotation = 2*pi/number_of_sides
        vertices = []
        faces = []
        faces.append(Face())

        faces[0].outer_vertices = self.generate_outer_vertices(number_of_sides=number_of_sides, initial_angle=initial_angle)
        faces[0].offset_outer_vertices(offset=glm.vec3([0.0, -1.0, 0.0]))
        faces[0].vertices = self.outer_vertices_to_vertices(number_of_sides=number_of_sides, face=faces[0])
        faces[0].apply_hardset_quad_texture_coords()

        #generate the second face via exude
        faces.append(Face())
        faces[1].extrude_from_other_face(other=faces[0], direction = [0.0, 1.0, 0.0], distance=self.dimensions[1])

        #generate the 4 vertical faces
        faces += self.stitch_faces(face_a=faces[0], face_b=faces[1], number_of_faces=number_of_sides)

        #generate the top of the wall
        x_border = 0.0
        z_border = 0.5
        y_border = 2.0
        y_wall_bottom = self.dimensions[1]
        top_of_wall = Face()
        bottom_of_wall = Face()

        #TODO: refactor this big time
        top_of_wall.outer_vertices = faces[0].outer_vertices.copy()
        top_of_wall.outer_vertices[0] -= x_border
        top_of_wall.outer_vertices[1] += y_border
        top_of_wall.outer_vertices[2] += z_border
        top_of_wall.outer_vertices[3] += x_border
        top_of_wall.outer_vertices[4] += y_border
        top_of_wall.outer_vertices[5] += z_border
        top_of_wall.outer_vertices[6] += x_border
        top_of_wall.outer_vertices[7] += y_border
        top_of_wall.outer_vertices[8] -= z_border
        top_of_wall.outer_vertices[9] -= x_border
        top_of_wall.outer_vertices[10] += y_border
        top_of_wall.outer_vertices[11] -= z_border

        top_of_wall.vertices = self.outer_vertices_to_vertices(number_of_sides=4, face=top_of_wall)
        # top_of_wall.apply_hardset_quad_texture_coords()
        faces.append(top_of_wall)

        bottom_of_wall.outer_vertices = faces[0].outer_vertices.copy()
        bottom_of_wall.outer_vertices[0] -= x_border
        bottom_of_wall.outer_vertices[1] += y_wall_bottom
        bottom_of_wall.outer_vertices[2] += z_border
        bottom_of_wall.outer_vertices[3] += x_border
        bottom_of_wall.outer_vertices[4] += y_wall_bottom
        bottom_of_wall.outer_vertices[5] += z_border
        bottom_of_wall.outer_vertices[6] += x_border
        bottom_of_wall.outer_vertices[7] += y_wall_bottom
        bottom_of_wall.outer_vertices[8] -= z_border
        bottom_of_wall.outer_vertices[9] -= x_border
        bottom_of_wall.outer_vertices[10] += y_wall_bottom
        bottom_of_wall.outer_vertices[11] -= z_border

        bottom_of_wall.vertices = self.outer_vertices_to_vertices(number_of_sides=4, face=bottom_of_wall)

        #wall vertical quads
        faces += self.stitch_faces(face_a=bottom_of_wall, face_b=top_of_wall, number_of_faces=number_of_sides)

        vertices = []
        beveled_faces = []
        for face in faces:
            beveled_faces += self.bevel_cut(face, bevel_depths=[0.0, -0.2], border_sizes=[0.6, 0.2])
        for face in beveled_faces:
            vertices += face.face_to_list()
        # for face in faces:
        #     vertices += face.face_to_list()
        return np.array(vertices,
            dtype=np.float32
        ).flatten()

class DoorwayTileMesh(FloorTileMesh):
    """
    straight walls for wfc world
    """
    def generate_vertices(self):
        radius = self.dimensions[0]
        number_of_sides = 4.0
        initial_angle = pi/number_of_sides

        angle_of_rotation = 2*pi/number_of_sides
        vertices = []
        faces = []
        faces.append(Face())
        faces[0].outer_vertices = self.generate_outer_vertices(number_of_sides=number_of_sides, initial_angle=initial_angle)
        faces[0].offset_outer_vertices(offset=glm.vec3([0.0, -1.0, 0.0]))
        faces[0].vertices = self.outer_vertices_to_vertices(number_of_sides=number_of_sides, face=faces[0])

        #generate the second face via exude
        faces.append(Face())
        faces[1].extrude_from_other_face(other=faces[0], direction = [0.0, 1.0, 0.0], distance=self.dimensions[1])

        #generate the 4 vertical faces
        faces += self.stitch_faces(face_a=faces[0], face_b=faces[1], number_of_faces=number_of_sides)

        #generate the top of the wall
        x_border = 0.0
        z_border = 0.5
        y_border = 2.0
        y_wall_bottom = self.dimensions[1]
        top_of_wall = Face()
        bottom_of_wall = Face()

        #TODO: refactor this big time
        top_of_wall.outer_vertices = faces[0].outer_vertices.copy()
        top_of_wall.outer_vertices[0] -= x_border
        top_of_wall.outer_vertices[1] += y_border
        top_of_wall.outer_vertices[2] += z_border
        top_of_wall.outer_vertices[3] += x_border
        top_of_wall.outer_vertices[4] += y_border
        top_of_wall.outer_vertices[5] += z_border
        top_of_wall.outer_vertices[6] += x_border
        top_of_wall.outer_vertices[7] += y_border
        top_of_wall.outer_vertices[8] -= z_border
        top_of_wall.outer_vertices[9] -= x_border
        top_of_wall.outer_vertices[10] += y_border
        top_of_wall.outer_vertices[11] -= z_border

        top_of_wall.vertices = self.outer_vertices_to_vertices(number_of_sides=4, face=top_of_wall)
        faces.append(top_of_wall)

        bottom_of_wall.outer_vertices = faces[0].outer_vertices.copy()
        bottom_of_wall.outer_vertices[0] -= x_border
        bottom_of_wall.outer_vertices[1] += y_wall_bottom
        bottom_of_wall.outer_vertices[2] += z_border
        bottom_of_wall.outer_vertices[3] += x_border
        bottom_of_wall.outer_vertices[4] += y_wall_bottom
        bottom_of_wall.outer_vertices[5] += z_border
        bottom_of_wall.outer_vertices[6] += x_border
        bottom_of_wall.outer_vertices[7] += y_wall_bottom
        bottom_of_wall.outer_vertices[8] -= z_border
        bottom_of_wall.outer_vertices[9] -= x_border
        bottom_of_wall.outer_vertices[10] += y_wall_bottom
        bottom_of_wall.outer_vertices[11] -= z_border

        bottom_of_wall.vertices = self.outer_vertices_to_vertices(number_of_sides=4, face=bottom_of_wall)

        #wall vertical faces
        faces_vertical = []
        faces_vertical += self.stitch_faces(face_a=bottom_of_wall, face_b=top_of_wall, number_of_faces=number_of_sides)
        #? order of faces?
            #-need to to get the two z-facing faces (parallel to x-axis)
            #should be 0 and 2 index


        doorway_faces = []
        doorway_faces += self.subdivide_into_doorway(
            face_z_negative=faces_vertical[0],
            face_z_positive=faces_vertical[2],
            doorway_width=0.4,
            doorway_height=.8
        )
        #delete faces from which we cut a door
        del faces_vertical[0]
        del faces_vertical[1]

        #add all faces to list, and numpy-ify so opengl can draw
        faces += faces_vertical + doorway_faces
        vertices = []
        beveled_faces = []
        for face in faces:
            beveled_faces += self.bevel_cut(face, bevel_depths = [0.0, -0.2], border_sizes = [0.6, 0.2])
        for face in beveled_faces:
            vertices += face.face_to_list()
        # for face in faces:
        #     vertices += face.face_to_list()
        return np.array(vertices,
            dtype=np.float32
        ).flatten()

    def subdivide_into_vertical_thirds(self, face, center_width=0.4, remove_center=False, z_positive=False):
        """
        use: making doors/doorways
        divides a square face into 3 vertical rectangles.  returns 3 new faces
        :param center_width: width of center rectangle
        :return subfaces, a list of 3(2 for hole) faces
        """
        subfaces = [] #return
        center_face = Face()
        left_face = Face()
        right_face = Face()
        center_width_normalized = center_width
        center_width *= face.tile_width
        if z_positive:
            upper_left = glm.vec3(face.outer_vertices[0], face.outer_vertices[1], face.outer_vertices[2])
            lower_left = glm.vec3(face.outer_vertices[3], face.outer_vertices[4], face.outer_vertices[5])
            lower_right = glm.vec3(face.outer_vertices[6], face.outer_vertices[7], face.outer_vertices[8])
            upper_right = glm.vec3(face.outer_vertices[9], face.outer_vertices[10], face.outer_vertices[11])
        else:
            upper_left = glm.vec3(face.outer_vertices[0], face.outer_vertices[1], face.outer_vertices[2])
            lower_left = glm.vec3(face.outer_vertices[3], face.outer_vertices[4], face.outer_vertices[5])
            lower_right = glm.vec3(face.outer_vertices[6], face.outer_vertices[7], face.outer_vertices[8])
            upper_right = glm.vec3(face.outer_vertices[9], face.outer_vertices[10], face.outer_vertices[11])

        # center quad's outer vertices
        midpoint_horizontal = (lower_right.x + lower_left.x) / 2.0
        face_plane_z = lower_left.z
        face_upper_y = upper_left.y
        face_lower_y = lower_left.y
        center_lower_left = glm.vec3(midpoint_horizontal - center_width / 2.0, face_lower_y, face_plane_z)
        center_lower_right = glm.vec3(midpoint_horizontal + center_width / 2.0, face_lower_y, face_plane_z)
        center_upper_right = glm.vec3(midpoint_horizontal + center_width / 2.0, face_upper_y, face_plane_z)
        center_upper_left = glm.vec3(midpoint_horizontal - center_width / 2.0, face_upper_y, face_plane_z)

        if z_positive:
            center_face.outer_vertices = list(center_upper_left) + list(center_lower_left) +\
                                         list(center_lower_right) + list(center_upper_right)
        else:
            center_face.outer_vertices = list(center_upper_right) + list(center_lower_right) + list(center_lower_left) + list(
                center_upper_left)

        center_face.vertices = center_face.outer_vertices_to_vertices()
        border_width = (1.0 - center_width_normalized) / 2
        center_face.set_texture_offsets(horizontal=[border_width, -border_width], vertical=[0.0,0.0])
        center_face.apply_hardset_quad_texture_coords()
        if not remove_center:
            subfaces.append(center_face)
            # subfaces += self.subdivide_into_horizontal_halves(center_face, bottom_height=0.9, z_positive=False)

        # now make the 2 border faces
        #left
        if z_positive:
            left_face.outer_vertices = list(upper_left)  + list(lower_left)  + \
                                       list(center_lower_left) + list(center_upper_left)
            left_face.vertices = left_face.outer_vertices_to_vertices()
            left_face.set_texture_offsets(horizontal=[0, -(border_width + center_width_normalized)], vertical=[0.0, 0.0])
            left_face.apply_hardset_quad_texture_coords()
        else: #right
            left_face.outer_vertices =  list(center_upper_left)  + list(center_lower_left) +\
                                         list(lower_right) + list(upper_right)
            left_face.vertices = left_face.outer_vertices_to_vertices()
            left_face.set_texture_offsets(horizontal=[border_width + center_width_normalized, 0], vertical=[0.0, 0.0])
            left_face.apply_hardset_quad_texture_coords()
        subfaces.append(left_face)

        #right
        if z_positive:
            right_face.outer_vertices = list(center_upper_right) + list(center_lower_right)  + \
                                         list(lower_right) + list(upper_right)
            right_face.vertices = right_face.outer_vertices_to_vertices()
            right_face.set_texture_offsets(horizontal=[border_width + center_width_normalized, 0], vertical=[0.0, 0.0])
            right_face.apply_hardset_quad_texture_coords()
        else: #left
            right_face.outer_vertices = list(upper_left)+ list(lower_left) +\
                                         list(center_lower_right)+ list(center_upper_right)
            right_face.vertices = right_face.outer_vertices_to_vertices()
            right_face.set_texture_offsets(horizontal=[0, -(border_width + center_width_normalized)], vertical=[0.0, 0.0])
            right_face.apply_hardset_quad_texture_coords()
        subfaces.append(right_face)


        return subfaces

    def subdivide_into_horizontal_halves(self, face, bottom_height=0.5, z_positive=False):
        """
        use: making doors/doorways
        divides a square face into 2 quad faces. returns 2 new faces
        :param bottom_height: height of lower rectangle
        :return subfaces, a list of 2 faces
        """
        subfaces = [] #return
        bottom_face = Face()
        top_face = Face()
        bottom_height_normalized = bottom_height
        # bottom_height *= face.tile_width
        # if z_positive:
        upper_right = glm.vec3(face.outer_vertices[0], face.outer_vertices[1], face.outer_vertices[2])
        lower_right = glm.vec3(face.outer_vertices[3], face.outer_vertices[4], face.outer_vertices[5])
        lower_left = glm.vec3(face.outer_vertices[6], face.outer_vertices[7], face.outer_vertices[8])
        upper_left = glm.vec3(face.outer_vertices[9], face.outer_vertices[10], face.outer_vertices[11])

        bottom_height = bottom_height*(upper_right.y - lower_right.y)
        # else:
        #     lower_right = glm.vec3(face.outer_vertices[0], face.outer_vertices[1], face.outer_vertices[2])
        #     upper_right = glm.vec3(face.outer_vertices[3], face.outer_vertices[4], face.outer_vertices[5])
        #     upper_left = glm.vec3(face.outer_vertices[6], face.outer_vertices[7], face.outer_vertices[8])
        #     lower_left = glm.vec3(face.outer_vertices[9], face.outer_vertices[10], face.outer_vertices[11])



        #middle line points
        middle_left = lower_left + glm.vec3(0.0, bottom_height, 0.0)
        middle_right = lower_right + glm.vec3(0.0, bottom_height, 0.0)

        #bottom quads outer vertices
        bottom_lower_left = lower_left
        bottom_lower_right = lower_right
        bottom_upper_left = middle_left
        bottom_upper_right = middle_right

        if z_positive:
            bottom_face.outer_vertices = list(bottom_upper_left) + list(bottom_lower_left) + \
                                         list(bottom_lower_right) + list(bottom_upper_right)
        else:
            bottom_face.outer_vertices = list(bottom_upper_right) + list(bottom_lower_right) + \
                                         list(bottom_lower_left) + list(bottom_upper_left)

        bottom_face.vertices = bottom_face.outer_vertices_to_vertices()
        bottom_face.set_texture_offsets(horizontal=face.texture_horizontal_offset,
                                        vertical=[0.0, -(1.0-bottom_height_normalized)])
        bottom_face.apply_hardset_quad_texture_coords()
        subfaces.append(bottom_face)

        # top quads outer vertices
        top_lower_left =  middle_left
        top_lower_right = middle_right
        top_upper_left = upper_left
        top_upper_right = upper_right

        if z_positive:
            top_face.outer_vertices = list(top_upper_left) + list(top_lower_left) + \
                                         list(top_lower_right) + list(top_upper_right)
        else:
            top_face.outer_vertices = list(top_upper_right) + list(top_lower_right) + \
                                         list(top_lower_left) + list(top_upper_left)

        top_face.vertices = top_face.outer_vertices_to_vertices()
        top_face.set_texture_offsets(horizontal=face.texture_horizontal_offset,
                                     vertical=[bottom_height_normalized, 0.0])
        top_face.apply_hardset_quad_texture_coords()
        subfaces.append(top_face)

        return subfaces

    def subdivide_into_doorway(self, face_z_positive, face_z_negative, doorway_width=0.5, doorway_height=0.8):
        """
        Takes 2 parallel faces and returns the 9 faces of a doorway

        TODO:
        just make it take 2 faces in and return one with the doorway-cut
        :param face_z_positive: face with normal towards z+
        :param face_z_negative: face with normal towards z-
        :param center_width: width of door
        :return:
        """
        faces = []
        third_faces_z_positive = self.subdivide_into_vertical_thirds(
            face=face_z_positive,
            center_width=doorway_width,
            remove_center=False,
            z_positive=True
        )
        faces.append(third_faces_z_positive[1]) #left of door
        faces.append(third_faces_z_positive[2]) #right of door
        door_faces_z_positive = self.subdivide_into_horizontal_halves(
            face=third_faces_z_positive[0],
            bottom_height=doorway_height,
            z_positive=False
        ) #subdivide the center face
        faces.append(door_faces_z_positive[1]) #above door

        third_faces_z_negative = self.subdivide_into_vertical_thirds(
            face=face_z_negative,
            center_width=doorway_width,
            remove_center=False,
            z_positive=False
        )
        faces.append(third_faces_z_negative[1])  # left of door
        faces.append(third_faces_z_negative[2])  # right of door
        door_faces_z_negative = self.subdivide_into_horizontal_halves(
            face=third_faces_z_negative[0],
            bottom_height=doorway_height,
            z_positive=False
        )  # subdivide the center face
        faces.append(door_faces_z_negative[1])  # above door

        #left side of the inner door hallway
        door_hole_z_negative = door_faces_z_negative[0].outer_vertices
        door_hole_z_positive = door_faces_z_positive[0].outer_vertices

        #get our outer vertices z+
        z_positive_upper_left = glm.vec3(door_hole_z_positive[0], door_hole_z_positive[1], door_hole_z_positive[2])
        z_positive_lower_left = glm.vec3(door_hole_z_positive[3], door_hole_z_positive[4], door_hole_z_positive[5])
        z_positive_lower_right = glm.vec3(door_hole_z_positive[6], door_hole_z_positive[7], door_hole_z_positive[8])
        z_positive_upper_right = glm.vec3(door_hole_z_positive[9], door_hole_z_positive[10], door_hole_z_positive[11])

        # get our outer vertices z-
        z_negative_upper_right = glm.vec3(door_hole_z_negative[0], door_hole_z_negative[1], door_hole_z_negative[2])
        z_negative_lower_right = glm.vec3(door_hole_z_negative[3], door_hole_z_negative[4], door_hole_z_negative[5])
        z_negative_lower_left = glm.vec3(door_hole_z_negative[6], door_hole_z_negative[7], door_hole_z_negative[8])
        z_negative_upper_left = glm.vec3(door_hole_z_negative[9], door_hole_z_negative[10], door_hole_z_negative[11])

        #construct left side
        left_hall = Face()
        left_hall.outer_vertices = list(z_positive_upper_left)  + list(z_positive_lower_left) + list(z_negative_lower_left) + list(z_negative_upper_left)
        left_hall.vertices = left_hall.outer_vertices_to_vertices()
        left_hall.set_texture_offsets(vertical=[0.0, 0.0], horizontal=[0.0, 0.0])
        left_hall.apply_hardset_quad_texture_coords()
        faces.append(left_hall)

        # construct left side
        right_hall = Face()
        right_hall.outer_vertices = list(z_negative_upper_right) + list(z_negative_lower_right) +\
                                    list(z_positive_lower_right) + list(z_positive_upper_right)
        right_hall.vertices = right_hall.outer_vertices_to_vertices()
        right_hall.set_texture_offsets(vertical=[0.0, 0.0], horizontal=[0.0, 0.0])
        right_hall.apply_hardset_quad_texture_coords()
        faces.append(right_hall)

        # construct ceiling
        ceiling_hall = Face()
        ceiling_hall.outer_vertices = list(z_negative_upper_right) + list(z_positive_upper_right) + \
                                    list(z_positive_upper_left) + list(z_negative_upper_left)
        ceiling_hall.vertices = ceiling_hall.outer_vertices_to_vertices()
        ceiling_hall.set_texture_offsets(vertical=[0.0, 0.0], horizontal=[0.0, 0.0])
        ceiling_hall.apply_hardset_quad_texture_coords()
        faces.append(ceiling_hall)

        return faces

class Prism(PrimativeMesh):
    """
    generates a prism based on an N-sided regular polygon.
    """

    def __init__(self,
                 shader,
                 # material properties
                 diffuse,
                 specular,
                 shininess=32.0,
                 # mesh properties
                 dimensions=[1.0, 1.0], #radius, height
                 position=[0.0, 0.0, 0.0],
                 rotation_magnitude=0,
                 rotation_axis=glm.vec3([0.0, 0.0, 1.0]),
                 scale=glm.vec3([1.0, 1.0, 1.0]),
                 sides=3
                 ):
        self.shader = shader
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.position = glm.vec3(position)
        self.rotation_magnitude = glm.vec3(rotation_magnitude)
        self.rotation_axis = glm.vec3(rotation_axis)
        self.scale = glm.vec3(scale)
        self.dimensions = dimensions
        self.texture_horizontal_offset = [0.0, 0.0]
        self.texture_vertical_offset = [0.0, 0.0]
        self.sides = sides
        self.vertices = self.generate_vertices()
        self.setup()

    def setup(self):
        # quad VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        # quad position vertices (vertex attribute)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # quad texture coords
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 8, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        # quad normals
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 8, ctypes.c_void_p(20))
        glEnableVertexAttribArray(2)

        self.model_loc = glGetUniformLocation(self.shader, "model")

    def generate_vertices(self):
        """
        Generates the outer vertices and vertices of the triangle of this model
        """
        radius = self.dimensions[0]
        height = 0
        try:
            height = self.dimensions[1]
        except IndexError:
            height = math.sqrt(radius * radius * 2)
        number_of_sides = self.sides
        initial_angle = pi / number_of_sides
        angle_of_rotation = 2 * pi / number_of_sides
        faces = [self._generate_polygonal_face(number_of_sides=number_of_sides)]
        # generate the second face via exude
        faces.append(Face())
        faces[1].extrude_from_other_face(other=faces[0], direction=[0.0, 1.0, 0.0], distance=height)
        # generate the 4 vertical faces
        faces += self.stitch_faces(face_a=faces[0], face_b=faces[1], number_of_faces=number_of_sides)
        vertices = []
        for face in faces:
            vertices += face.face_to_list()
        return np.array(
            vertices,
            dtype=np.float32
        ).flatten()

class NPrismBezierCut(Prism):
    def generate_vertices(self):
        """
        Testing our bezier cut by applying it to an N-sided Prisms quad faces
        """
        radius = self.dimensions[0]
        height = 0
        try:
            height = self.dimensions[1]
        except IndexError:
            height = math.sqrt(radius * radius * 2)
        number_of_sides = self.sides
        initial_angle = pi / number_of_sides
        angle_of_rotation = 2 * pi / number_of_sides
        faces = [Face()]
        faces[0].outer_vertices = self.generate_outer_vertices(number_of_sides=number_of_sides,                                                           initial_angle=initial_angle)
        # faces[0].offset_outer_vertices(offset=glm.vec3([0.0, -1.0, 0.0]))
        faces[0].vertices = self.outer_vertices_to_vertices(number_of_sides=number_of_sides, face=faces[0])
        # # faces[0].apply_hardset_quad_texture_coords()
        # # generate the second face via exude
        faces.append(Face())
        faces[1].extrude_from_other_face(other=faces[0], direction=[0.0, 1.0, 0.0], distance=height)
        # # generate the 4 vertical faces
        faces += self.stitch_faces(face_a=faces[0], face_b=faces[1], number_of_faces=number_of_sides)
        # quad_faces = self.divide_face_into_quads(original_face=faces[2], center_point_offset=0.0)
        quad_faces = self.subdivide_quad_lengthwise(face=faces[2], subdivisions=3, widthwise=True)
        for face in quad_faces:
            faces_bevel_cut = self.bevel_cut(original_face=face,bevel_depths=[-0.0,-1.5], border_sizes=[0.1,0.3], depth=2, direction=glm.vec3(.5,0.0,-1.0))
            faces_from_bezier_cut = self.bezier_cut(face=faces_bevel_cut[-5], intervals=8, offset=0.2)
            del faces_bevel_cut[-5]
            for face in faces_from_bezier_cut:
                faces.insert(3, face)
            for face in faces_bevel_cut:
                faces.insert(3, face)
        del faces[2]

        hip_faces = self.subdivide_hip_roof(faces[-1], start_point=[0.1, 0.1], end_point=[0.1, 0.9], vertical_offset=1.0)
        del faces[-1]
        for face in hip_faces:
            faces.insert(3, face)

        vertices = []
        index = 0
        # faces[-3].update_texture_coords_using_atlas_index(texture_atlas_index=3, texture_atlas_size=2)
        for face in faces:
            face.update_texture_coords_using_atlas_index(texture_atlas_index=index%4, texture_atlas_size=2)
            vertices += face.face_to_list()
            index += 1
        return np.array(vertices,
                        dtype=np.float32
        ).flatten()

class SegmentedPrism(Prism):
    """
    Makes potion bottles (segmented cylindar with different radius per segment)
    """
    def __init__(self,
        shader,
        # material properties
        diffuse,
        specular,
        shininess=32.0,
        # mesh properties
        dimensions=[1.0, 1.0], #radius, height
        position=[0.0, 0.0, 0.0],
        rotation_magnitude=0,
        rotation_axis=glm.vec3([0.0, 0.0, 1.0]),
        scale=glm.vec3([1.0, 1.0, 1.0]),
        sides=3,
        segments=1,
        outer_vertices_offset = [0.0, -1.0, 0.0],
        bevel_depths = [-0.2, -0.4],
        border_sizes = [0.6, 0.0],
        depth=2
    ):
        self.shader = shader
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.position = glm.vec3(position)
        self.rotation_magnitude = glm.vec3(rotation_magnitude)
        self.rotation_axis = glm.vec3(rotation_axis)
        self.scale = glm.vec3(scale)
        self.dimensions = dimensions
        self.texture_horizontal_offset = [0.0, 0.0]
        self.texture_vertical_offset = [0.0, 0.0]
        self.sides = sides
        self.segments = segments
        self.bevel_depths = bevel_depths
        self.border_sizes = border_sizes
        self.depth = depth
        self.outer_vertices_offset = outer_vertices_offset
        self.vertices = self.generate_vertices()
        self.setup()


    def generate_vertices(self):

        radius = self.dimensions[0]
        height = self.dimensions[1]
        segment_height = height/self.segments
        number_of_sides = self.sides
        initial_angle = pi / number_of_sides

        angle_of_rotation = 2 * pi / number_of_sides
        vertices = []
        faces = []
        faces.append(Face())

        faces[0].outer_vertices = self.generate_outer_vertices(number_of_sides=number_of_sides,
                                                               initial_angle=initial_angle)
        faces[0].offset_outer_vertices(offset=glm.vec3(self.outer_vertices_offset))
        faces[0].vertices = self.outer_vertices_to_vertices(number_of_sides=number_of_sides, face=faces[0])

        # generate the second face via exude
        base_faces = [faces[0]]
        radius = 1.0
        for index in range(1, self.segments+1):
            base_faces.append(Face())

            if index < self.segments/4:
                radius *= 1.0 + 0.1*index
            else:
                radius *= 0.9 #+ .1*index
            base_faces[index].extrude_from_other_face(other=base_faces[index - 1], direction=[0.0, 1.0, 0.0], distance=segment_height, radius=radius)
            faces += self.stitch_faces(face_a=base_faces[index-1], face_b=base_faces[index], number_of_faces=number_of_sides)
            if index == self.segments:
                faces.append(base_faces[index])

        vertices = []
        for face in faces:
            vertices += face.face_to_list()
        return np.array(vertices,
                        dtype=np.float32
                        ).flatten()

class SegmentedPrismBevelCutTest(SegmentedPrism):
    """
    testing our bevel cut on an a multi-faced, non-axis-aligned mesh based
    """
    def generate_vertices(self):

        radius = self.dimensions[0]
        height = self.dimensions[1]
        segment_height = height/self.segments
        number_of_sides = self.sides
        initial_angle = pi / number_of_sides

        angle_of_rotation = 2 * pi / number_of_sides
        vertices = []
        faces = []
        faces.append(Face())

        faces[0].outer_vertices = self.generate_outer_vertices(number_of_sides=number_of_sides,
                                                               initial_angle=initial_angle)
        faces[0].offset_outer_vertices(offset=glm.vec3(self.outer_vertices_offset))
        faces[0].vertices = self.outer_vertices_to_vertices(number_of_sides=number_of_sides, face=faces[0])

        # generate the later N faces via exude
        base_faces = [faces[0]]
        radius = 1.0
        for index in range(1, self.segments+1):
            base_faces.append(Face())

            if index < self.segments/4:
                radius *= 1.0 + 0.1*index
            else:
                radius *= 0.9 #+ .1*index
            base_faces[index].extrude_from_other_face(other=base_faces[index - 1], direction=[0.0, 1.0, 0.0], distance=segment_height, radius=radius)
            base_faces[index].radius = radius
            faces += self.stitch_faces(face_a=base_faces[index-1], face_b=base_faces[index], number_of_faces=number_of_sides)
            if index == self.segments:
                faces.append(base_faces[index])

        vertices = []
        #apply bevel cut
        bevel_cut_faces = []
        # face_to_bevel_cut = faces.pop(len(faces)-1)
        # bevel_cut_faces = self.bevel_cut(face_to_bevel_cut)
        # faces += bevel_cut_faces
        faces[0].calculate_outer_vertices_as_vec3()
        faces[0].flip_winding_order()

        for index, face in enumerate(faces):
            face_to_cut = faces[index]
            bevel_cut_faces += self.bevel_cut(
                face_to_cut,
                bevel_depths=self.bevel_depths,
                border_sizes=self.border_sizes,
                depth=self.depth,
            )
        for face in bevel_cut_faces:
            vertices += face.face_to_list()
        return np.array(vertices,
                        dtype=np.float32
                        ).flatten()

class SegmentedPrismBevelPolygonCornerTest(SegmentedPrism):
    """
    using our bevel cut function (bevel_polygon_corner) on a base polygon.
    Also testing the divide_face_into_quads
    """
    def generate_vertices(self):
        radius = self.dimensions[0]
        height = self.dimensions[1]
        segment_height = height/self.segments
        initial_angle = pi / self.sides
        angle_of_rotation = 2 * pi / self.sides
        vertices = []
        faces = []
        faces.append(Face())
        faces[0].outer_vertices = self.generate_outer_vertices(
            number_of_sides=self.sides,
            initial_angle=initial_angle
        )
        faces[0].offset_outer_vertices(offset=glm.vec3(self.outer_vertices_offset))
        """NEW beveling on base polygon testing-working"""
        for i in range(1):
            faces[0] = self.bevel_polygon_corner(faces[0], subject_vertex_index=0, bevel_ratio=0.25)
        for i in range(1):
            faces[0] = self.bevel_polygon_corner(faces[0], subject_vertex_index=3, bevel_ratio=0.25)
        faces[0].vertices = self.outer_vertices_to_vertices(number_of_sides=self.sides, face=faces[0])
        # generate the later N faces via exude
        base_faces = [faces[0]]
        radius = 1.0
        for index in range(1, self.segments+1):
            base_faces.append(Face())
            if index < self.segments/4:
                radius *= 1.0 + 0.1*index
            else:
                radius *= 0.9 #+ .1*index
            base_faces[index].extrude_from_other_face(other=base_faces[index - 1], direction=[0.0, 1.0, 0.0], distance=segment_height, radius=radius)
            base_faces[index].radius = radius
            faces += self.stitch_faces(face_a=base_faces[index-1], face_b=base_faces[index], number_of_faces=self.sides)
            if index == self.segments:
                faces.append(base_faces[index])
        """NEW quad subdivide of faces function testing"""
        faces_post_divide = []
        faces[0].calculate_outer_vertices_as_vec3()
        faces[0].flip_winding_order()
        for face in faces:
            # faces_post_divide += self.divide_face_into_quads(face, center_point_offset=0.2)
            faces_post_divide += self.pyrimidize_face(face, center_point_offset=0.2)
        faces = faces_post_divide
        vertices = []
        bevel_cut_faces = []
        faces[0].calculate_outer_vertices_as_vec3()
        faces[0].flip_winding_order()
        for index, face in enumerate(faces):
            face_to_cut = faces[index]
            bevel_cut_faces += self.bevel_cut(
                face_to_cut,
                bevel_depths = self.bevel_depths,
                border_sizes = self.border_sizes,
                depth=self.depth,
            )
        for face in bevel_cut_faces:
            vertices += face.face_to_list()
        return np.array(vertices,
                        dtype=np.float32
        ).flatten()

def calculate_normal(vertices=[]):
    """
    calculates the normal vector
    :param vertices: a list of 3 glm vec3's
    :return: a vec3 with the normal to this triangle
    """
    return glm.normalize(glm.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]))

def calculate_mean_point(points):
    """
    Calculate the mean point of a polygon
    :param points: list of points of a polygon as glm.vec3s
    :return: mean, a glm.vec3
    """
    sum_point = glm.vec3([0.0, 0.0, 0.0])
    for point in points:
        sum_point += point
    return sum_point / len(points)

def list_to_vec3_list(current_list):
    """
    convert a list of points (serialized fully, as a list) to a list of glm.vec3's
    :param current_list:
    :return:
    """
    reformated_list = []
    index = 0
    while index < len(current_list):
        reformated_list.append(
            glm.vec3(
                current_list[index],
                current_list[index + 1],
                current_list[index + 2],
            )
        )
        index += 3
    return reformated_list

def vec3_list_to_list(vec3_list):
    """
    convert a list of points-as-vec3's to a serialized list
    :return: a list
    """
    serial_list = []
    for point in vec3_list:
        serial_list += list(point)
    return serial_list

def bezier_linear(points=[], intervals=0):
    """
    Linearly interpolate between two points
    :param points: two points to interpolate between.  glm.vec3
    :param intervals: number of steps to take between point_0 and point_1
    :return: a list of points (including start and end control points)
    """
    bezier_points = points.copy()
    for step in range(1,intervals+1):
        time = step*(1.0/intervals)
        new_point = points[0] + time*(points[1]-points[0])
        bezier_points.insert(-1, new_point)
    return bezier_points

def bezier_quadratic(points=[], intervals=0):
    """
    Quadratically interpolate between two points
    :param points: three points to interpolate between.  glm.vec3
    :param intervals: number of steps to take between point_0 and point_1
    :return: a list of points
    """
    bezier_points = [points[0], points[2]]
    control_points = points.copy()
    for step in range(1, intervals + 1):
        time = step * (1.0 / (intervals + 1))
        new_point = control_points[1] + pow((1.0 - time), 2) * (control_points[0] - control_points[1]) + pow(time,2) * \
                    (control_points[2] - control_points[1])
        bezier_points.insert(-1, new_point)
    return bezier_points

def bezier_cubic(points=[], intervals=0):
    """
        cubically interpolate between two points
        :param points: three points to interpolate between.  glm.vec3
        :param intervals: number of steps to take between point_0 and point_1
        :return: a list of points

    EXAMPLE
    bezier_points = [
    glm.vec3(0.0, 10.0, 0.0),
    glm.vec3(0.0, 25.0, 0.0),
    glm.vec3(45.0, 25.0, 0.0),
    glm.vec3(45.0, 10.0, 0.0)
    ]
    bezier_points = primatives.bezier_cubic(points=bezier_points, intervals=20)

    IN DRAW LOOP:
        for point in bezier_points:
            bezier_model.position = point
            bezier_model.draw(view=view)
    """
    bezier_points = [points[0], points[3]]
    control_points = points.copy()
    fractional_interval = 1.0/(intervals + 1)
    for step in range(1, intervals + 1):
        time = step*(fractional_interval)
        reverse_time = 1.0 - time
        new_point = pow(reverse_time, 3)*points[0] + 3.0 * pow(reverse_time, 2) * time * points[1] + \
                    3.0*reverse_time*pow(time,2)*points[2] + pow(time,3)*points[3]
        bezier_points.insert(-1, new_point)
    return bezier_points

def calculate_texture_coordinates_from_control_points(point, control_points):
    """
    given the 4 control points, calculate the proper texture coord.
    Assumes the 4 control points fall on exact corners of the texture
    :return: glm.vec2(width, height), each in range [0,1]
    """
    control_points = rotate_list(control_points, 2)
    height = glm.distance(control_points[0], control_points[1])
    width = glm.distance(control_points[1], control_points[2])
    height_delta = calculate_triangle_height(points=[control_points[2], point, control_points[1]])
    width_delta = calculate_triangle_height(points=[control_points[0], point, control_points[1]])
    texture_coordinates = glm.vec2(
        glm.clamp(1.0 - width_delta/width, 0.0, 1.0),
        glm.clamp(height_delta/height, 0.0, 1.0)
    )
    return texture_coordinates

def calculate_triangle_height(points = []):
    """
    Given 3 points of a triangle, return the height
    The points must be given counter-clockwise, with the first and third point forming the base
    TODO: Refactor
    :param points:
    :return: the height of the triangle
    """
    triangle_sides = triangle_points_to_side_lengths(points=points)
    area = herons_formula(triangle_sides)
    base = glm.distance(points[0], points[2])
    height = 2.0*area/base
    return height

def triangle_points_to_side_lengths(points):
    """
    Given the 3 points of a triangle, calculate the lengths of the 3 sides
    :param points:
    :return: list of sides
    """
    sides = []
    sides.append(glm.distance(points[0], points[1]))
    sides.append(glm.distance(points[1], points[2]))
    sides.append(glm.distance(points[2], points[0]))
    return sides

def herons_formula(sides):
    """
    Uses Heron's Formula to calculate the area of a triangle given the three sides
    :param sides:
    :return: Area of the triangle
    """
    return 0.25 * math.sqrt(4.0 * sides[0]*sides[0]*sides[1]*sides[1] - pow(sides[0]*sides[0] + sides[1]*sides[1] - \
                                                                            sides[2]*sides[2], 2))

def rotate_list(current_list, steps):
    """
    rotates a list by n steps.  negative steps puts last element first.
    EXAMPLE
    fruits = ['a', 'b', 'c']
    updated_fruits = rotate(fruits, -1)
    updated_fruits is now ['c', 'b', 'a']
    :param current_list: list to operate on.  Original not changed
    :param steps: number of elements to shift over
    :return: a new list
    """
    return current_list[steps:] + current_list[:steps]

def get_next_index(size, index):
    if index + 1 > size - 1:
        return 0
    else:
        return index + 1

def get_previous_index(size, index):
    if index - 1 < 0:
        return size - 1
    else:
        return index - 1