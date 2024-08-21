# class id2name for mapillary_vistas dataset
SEG_ID2NAME = ['Bird', 'Ground Animal', 'Curb', 'Fence', 'Guard Rail', 
                 'Barrier', 'Wall', 'Bike Lane', 'Crosswalk - Plain', 'Curb Cut', 
                 'Parking', 'Pedestrian Area', 'Rail Track', 'Road', 'Service Lane', 
                 'Sidewalk', 'Bridge', 'Building', 'Tunnel', 'Person', 'Bicyclist', 
                 'Motorcyclist', 'Other Rider', 'Lane Marking - Crosswalk', 'Lane Marking - General',
                 'Mountain', 'Sand', 'Sky', 'Snow', 'Terrain', 'Vegetation', 'Water', 'Banner', 
                 'Bench', 'Bike Rack', 'Billboard', 'Catch Basin', 'CCTV Camera', 'Fire Hydrant', 
                 'Junction Box', 'Mailbox', 'Manhole', 'Phone Booth', 'Pothole', 'Street Light', 
                 'Pole', 'Traffic Sign Frame', 'Utility Pole', 'Traffic Light', 'Traffic Sign (Back)', 
                 'Traffic Sign (Front)', 'Trash Can', 'Bicycle', 'Boat', 'Bus', 'Car', 'Caravan', 
                 'Motorcycle', 'On Rails', 'Other Vehicle', 'Trailer', 'Truck', 'Wheeled Slow', 'Car Mount', 'Ego Vehicle']

SEG_NAME2ID = {}
for i in range(len(SEG_ID2NAME)):
    SEG_NAME2ID[SEG_ID2NAME[i]] = i

ANIMAL = ['Bird', 'Ground Animal']
SUPPORT = ['Pole', 'Traffic Sign Frame', 'Utility Pole']
VEHICLE = ['Bicycle', 'Boat', 'Bus', 'Car', 'Caravan', 
        'Motorcycle', 'On Rails', 'Other Vehicle', 'Trailer', 'Truck', 'Wheeled Slow']
TRAFFIC_SIGN = ['Traffic Sign (Back)', 'Traffic Sign (Front)']
FLAT = ['Bike Lane', 'Crosswalk - Plain', 'Curb Cut', 
        'Parking', 'Pedestrian Area', 'Rail Track', 'Road', 'Service Lane', 'Sidewalk']
BARRIER = ['Curb', 'Fence', 'Guard Rail', 'Barrier', 'Wall']
STRUCTURE = ['Bridge', 'Building', 'Tunnel']
HUMAN = ['Person', 'Bicyclist', 'Motorcyclist', 'Other Rider']
NATURE = ['Mountain', 'Sand', 'Snow', 'Terrain', 'Vegetation', 'Water']
MARKING = ['Lane Marking - Crosswalk', 'Lane Marking - General']
VOID = ['Car Mount', 'Ego Vehicle']
OTHER_OBJECT = ['Banner', 'Bench', 'Bike Rack', 'Billboard', 'Catch Basin', 'CCTV Camera', 'Fire Hydrant', 
                 'Junction Box', 'Mailbox', 'Manhole', 'Phone Booth', 'Pothole', 'Street Light', 'Trash Can']
TRAFFIC_LIGHT = ['Traffic Light']
SKY = ['Sky']

DYNAMIC_OBJECT = HUMAN + ANIMAL # EXCEPT VEHICLE
STATIC_OBJECT = SUPPORT + TRAFFIC_SIGN + BARRIER + STRUCTURE + NATURE + MARKING + VOID + OTHER_OBJECT + TRAFFIC_LIGHT # EXCEPT FLAT and SKY
assert len(DYNAMIC_OBJECT + STATIC_OBJECT + FLAT + VEHICLE + SKY) == len(SEG_ID2NAME)

VEHICLE_ID = [SEG_NAME2ID[cls_name] for cls_name in VEHICLE]
DYNAMIC_OBJECT_ID = [SEG_NAME2ID[cls_name] for cls_name in DYNAMIC_OBJECT]
FLAT_ID = [SEG_NAME2ID[cls_name] for cls_name in FLAT]
STATIC_OBJECT_ID = [SEG_NAME2ID[cls_name] for cls_name in STATIC_OBJECT]

class THING():
    STATIC_OBJECT = 0
    DYNAMIC_OBJECT = 1
    ROAD = 2
    SKY = 3
    VEHICLE = 4


