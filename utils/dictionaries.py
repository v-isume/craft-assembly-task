#Collection of auxiliary dictionaries to process the meshes

#Part id to color dictionary
id_to_color = {
    0: [255, 255, 51, 255],
    1: [255, 0, 0, 255],
    2: [0, 255, 0, 255],
    3: [0, 0, 255, 255],
    4: [51, 51, 200, 255],
    5: [200, 51, 51, 255],
    6: [51, 200, 51, 255],
    7: [200, 51, 200, 255],
    8: [100, 100, 0, 255],
    9: [150, 150, 50, 255],
    10: [170, 255, 170, 255],
    11: [170, 170, 255, 255]
}

#Part type to part color dictionary
class_to_color = { 
    'truck_body':    [255, 255, 51, 255],       #truck_body
    'truck_cabin':   [255, 0, 255, 255],        #truck_cabin
    'wheel':         [0, 255, 255, 255],        #wheel
    'bus_body':      [255, 153, 51, 255],       #bus_body
    'furniture_leg': [255, 0, 0, 255],          #furniture_leg
    'chair_seat':    [0, 0, 255, 255],          #chair_seat
    'chair_back':    [0, 255, 0, 255],          #chair_back
    'chair_arm':     [0, 255, 128, 255],        #chair_arm
    'table_surface': [102, 102, 0, 255],        #table_surface
    'axle':          [255, 255, 255, 255]       #axle
}

#Part type to part color dictionary
classid_to_color = { 
    0: [255, 255, 51], #truck_body
    1: [255, 0, 255],  #truck_cabin
    2: [0, 255, 255],  #wheel
    3: [255, 153, 51], #bus_body
    4: [255, 0, 0],    #furniture_leg
    5: [0, 0, 255],    #chair_seat
    6: [0, 255, 0],    #chair_back
    7: [0, 255, 128],  #chair_arm
    8: [102, 102, 0]   #table_surface
}

#Primitive dictionary
primitive_dictionary = {
    'truck_body':       'cuboid',
    'truck_cabin':      'cuboid',
    'wheel':            'cylinder',
    'bus_body':         'cuboid',
    'furniture_leg':    'cuboid',
    'chair_seat':       'cuboid',
    'chair_back':       'cuboid',
    'chair_arm':        'cuboid',
    'table_surface':    'cuboid'
}

#Minimum template refers to a template mesh with the minimum amount of parts required for the target object.
#Is used as a reference in case the pose and symmetry fails to propose a minimum available object
minimum_template = {"bus" : "bus2_colored.obj", 
                    "chair" : "chair1_colored.obj",
                    "table" : "table2_colored.obj",
                    "truck" : "truck3_colored.obj"}

tp_gt = {
    0: 1,
    1: 2,
    2: 3,
    3: 3,
    4: 4,
    5: 1,
    6: 4,
    7: 3,
    8: 5,
    9: 6,
    10: 7,
    11: 8,
    12: 9,
    13: 8,
    14: 7,
    15: 10,
    16: 7,
    17: 7,
    18: 8,
    19: 8,
    20: 3
}

# tp_gt = {
#     0: 1,
#     1: 1,
#     2: 2,
#     3: 3,
#     4: 3,
#     5: 3,
#     6: 3,
#     7: 3,
#     8: 3,
#     9: 4,
#     10: 4,
#     11: 5,
#     12: 6,
#     13: 6,
#     14: 6,
#     15: 6,
#     16: 7,
#     17: 8
# }


class Primitive_scene():
    def __init__(self, dimensions, translation, rotation, shape):
        self.dimensions = dimensions
        self.translation = translation
        self.rotation = rotation
        self.shape = shape