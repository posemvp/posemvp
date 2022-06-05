import math


def find_angle(A, B, C):
    AB = math.sqrt(math.pow(B.x - A.x, 2) + math.pow(B.y - A.y, 2))
    BC = math.sqrt(math.pow(B.x - C.x, 2) + math.pow(B.y - C.y, 2))
    CA = math.sqrt(math.pow(C.x - A.x, 2) + math.pow(C.y - A.y, 2))
    return math.acos((BC * BC + AB * AB - CA * CA) / (2 * BC * AB)) * 180 / math.pi
