"""
Anything physics-related that we are coding ourself (verses using a library like pybullet)
"""


from glm import vec3, normalize, vec4, dot
import logging

logger = logging.getLogger(__name__)
logging.basicConfig()
# logging.root.setLevel(logging.NOTSET)

#Ray-plane intersection

#given
# ray_origin = vec3([10.0, 10.0, 10.0])
# ray_direction = normalize(vec3((0.0, -1.0, 0.0)))
# plane = vec4(0.0, 1.0, 0.0, 0.0)

# ray_origin = vec3((2.0, 3.0, 4.0))
# ray_direction = vec3((0.577, 0.577, 0.577))
# plane = vec4(1.0, 0.0, 0.0, -7.0)


def find_intersection_ray_plane(ray_origin, ray_direction, plane):
    """
    Find the intersection between a ray and plane
    created with reference to: https://education.siggraph.org/static/HyperGraph/raytrace/rayplane_intersection.htm
    :param ray_origin: origin of the ray
    :param ray_direction: NORMALIZED direction of the ray
    :param plane: defined as vec4(a,b,c,d), where abc are the normal, and d is the distance to (0,0,0)
    :return:
    """
    point_intersection = vec3(0.0, 0.0, 0.0)
    ray_direction = normalize(ray_direction)
    unit_normal = plane.xyz
    V_d = dot(unit_normal, ray_direction)
    if V_d == 0:
        logger.info('Ray is parallel to the plane.')
    elif V_d > 0:
        logger.info('Ray is pointing away from the plane.')
    else:
        V_0 = -(dot(unit_normal, ray_origin) + plane.w)
        t = V_0 / V_d
        if t < 0:
            logger.info('Ray intersects plane behind origin, ie, no intersection of interest')
        else:
            point_intersection = ray_origin + ray_direction * t
            logger.info(f'{point_intersection=}')
    return point_intersection

# find_intersection_ray_plane(ray_origin=ray_origin, ray_direction=ray_direction, plane=plane)

