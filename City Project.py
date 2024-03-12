from math import radians, sin, cos, sqrt, atan2
from collections import deque
from heapq import heappop, heappush
import time

def inputCity(prompt):
    city = input(prompt)
    while not(city in cities.keys()):
        print("City doesn't exist.")
        city = input(prompt)
    return city


def distance(start, end):
    return sqrt((cities[start]["coos"][0] - cities[end]["coos"][0])**2 + (cities[start]["coos"][1] - cities[end]["coos"][1])**2)
    return sqrt((cities[start]["coos"][0] - cities[end]["coos"][0])**2 + (cities[start]["coos"][1] - cities[end]["coos"][1])**2)

def breadth(start_city, end_city):
    if start_city not in cities.keys() or end_city not in cities.keys():
        raise ValueError("Start or end city not found in the graph.")

    visited = set()
    queue = deque([(start_city, [start_city])])

    while queue:
        current_city, path = queue.popleft()

        if current_city == end_city:
            return path

        if current_city not in visited:
            visited.add(current_city)

            for neighbor, distance in cities[current_city]["adjs"].items():
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))

    raise ValueError("No path found between the given cities.")

def depth(start_city, end_city):
    if start_city not in cities.keys() or end_city not in cities.keys():
        raise ValueError("Start or end city not found in the graph.")

    visited = set()
    shortest_path = None

    def dfs(current_city, path, current_distance):
        nonlocal shortest_path

        visited.add(current_city)

        if current_city == end_city:
            if shortest_path is None or current_distance < shortest_path[1]:
                shortest_path = (path, current_distance)
            return

        for neighbor, distance in cities[current_city]["adjs"].items():
            if neighbor not in visited:
                new_path = path + [neighbor]
                new_distance = current_distance + distance
                dfs(neighbor, new_path, new_distance)

    dfs(start_city, [start_city], 0)

    if shortest_path is not None:
        return shortest_path[0]

    raise ValueError("No path found between the given cities.")

def id_depth(start_city, end_city):
    if start_city not in cities.keys() or end_city not in cities.keys():
        raise ValueError("Start or end city not found in the graph.")

    def dfs_limit_depth(current_city, path, depth_limit):
        if depth_limit == 0:
            return None

        if current_city == end_city:
            return path

        for neighbor, distance in cities[current_city]["adjs"].items():
            new_path = path + [neighbor]
            result = dfs_limit_depth(neighbor, new_path, depth_limit - 1)
            if result:
                return result

        return None

    max_depth = 0

    while True:
        result = dfs_limit_depth(start_city, [start_city], max_depth)
        if result:
            return result
        max_depth += 1

def best_first(start_city, end_city):
    if start_city not in cities.keys() or end_city not in cities.keys():
        raise ValueError("Start or end city not found in the graph.")

    def heuristic(coord1, coord2):
        return haversine_distance(coord1, coord2)

    open_set = [(heuristic(cities[start_city]["coos"], cities[end_city]["coos"]), start_city)]
    closed_set = set()
    came_from = {}

    while open_set:
        _, current_city = heappop(open_set)

        if current_city == end_city:
            path = [current_city]
            while current_city in came_from:
                current_city = came_from[current_city]
                path.insert(0, current_city)
            return path

        closed_set.add(current_city)

        for neighbor, _ in sorted(cities[current_city]["adjs"].items(), key=lambda x: heuristic(cities[x[0]]["coos"], cities[end_city]["coos"])):
            if neighbor not in closed_set:
                heappush(open_set, (heuristic(cities[neighbor]["coos"], cities[end_city]["coos"]), neighbor))
                came_from[neighbor] = current_city

    raise ValueError("No path found between the given cities.")

def haversine_distance(coord1, coord2):
    # Radius of the Earth in kilometers
    R = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)
    
    # Calculate differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    # Calculate the distance
    distance = R * c
    
    return distance

def a_star(start_city, end_city):
    if start_city not in cities.keys() or end_city not in cities.keys():
        raise ValueError("Start or end city not found in the graph.")

    def heuristic(coord1, coord2):
        return haversine_distance(coord1, coord2)

    def reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.insert(0, current)
        return path

    open_set = [(0, start_city)]
    closed_set = set()
    came_from = {}
    g_score = {city: float('inf') for city in cities.keys()}
    g_score[start_city] = 0

    while open_set:
        current_cost, current_city = heappop(open_set)

        if current_city == end_city:
            return reconstruct_path(came_from, end_city)

        closed_set.add(current_city)

        for neighbor, distance in cities[current_city]["adjs"].items():
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current_city] + distance

            if tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                total_cost = tentative_g_score + heuristic(cities[neighbor]["coos"], cities[end_city]["coos"])
                heappush(open_set, (total_cost, neighbor))
                came_from[neighbor] = current_city

    raise ValueError("No path found between the given cities.")

def format_route(route):
    string = ''
    for i in range(len(route[:-1])):
        cityA = route[i]
        cityB = route[i + 1]
        string += cityA + " -" + "%0.2f" % cities[cityA]["adjs"][cityB] + "-> "

    string += route[-1]
    return string
cities = {}

coos = open("coordinates.csv").read().split("\n")
adjs = open("adjacencies.txt").read().replace(" \n","\n").split("\n")


for data in coos:
    if len(data.split(",")) == 3:
        city, coo1, coo2 = data.split(",")
        cities[city] = {
            "coos": [float(coo1), float(coo2)],
            "adjs": {}
        }
    elif data != "":
        print("Invalid coordinates:",data)
    
for pair in adjs:
    cityA, cityB = pair.split(" ")
    if cityA in cities.keys():
        if not(cityB in cities[cityA]["adjs"].keys()):
            cities[cityA]["adjs"][cityB] = haversine_distance(cities[cityA]["coos"], cities[cityB]["coos"])
    else:
        print("Error: no coords for:", cityA)
    if cityB in cities.keys():
        if not(cityA in cities[cityB]["adjs"].keys()):
            cities[cityB]["adjs"][cityA] = haversine_distance(cities[cityB]["coos"], cities[cityA]["coos"])
    else:
        print("Error: no coords for:", cityB)

while True:
    cityStart = inputCity("Starting city -> ")

    cityEnd = inputCity("Ending city -> ")

    print("""Undirected brute-force approaches:
[A] breadth-first search
[B] depth-first search
[C] ID-DFS search
Heuristic Approaches
[B] best-first search
[E] A* search""")
    method = input("Option [A], [B], [C], [D], or [E] -> ")
    while not(method.upper() in ["A","[A]", "B","[B]", "C","[C]", "D","[D]", "E","[E]"]):
        print("Unknown approach.")
        print("""Undirected brute-force approaches:
[A] breadth-first search
[B] depth-first search
[C] ID-DFS search
Heuristic Approaches:
[B] best-first search
[E] A* search""")
        method = input("Option [A], [B], [C], [D], or [E] -> ")


    if method.upper() == "A" or method.upper() == "[A]":
        start_time = time.time()
        print(format_route(breadth(cityStart, cityEnd)))
        end_time = time.time()
        print("%0.4f seconds" % (end_time-start_time))
    if method.upper() == "B" or method.upper() == "[B]":
        start_time = time.time()
        print(format_route(depth(cityStart, cityEnd)))
        end_time = time.time()
        print("%0.4f seconds" % (end_time-start_time))
    if method.upper() == "C" or method.upper() == "[C]":
        start_time = time.time()
        print(format_route(id_depth(cityStart, cityEnd)))
        end_time = time.time()
        print("%0.4f seconds" % (end_time-start_time))
    if method.upper() == "D" or method.upper() == "[D]":
        start_time = time.time()
        print(format_route(best_first(cityStart, cityEnd)))
        end_time = time.time()
        print("%0.4f seconds" % (end_time-start_time))
    if method.upper() == "E" or method.upper() == "[E]":
        start_time = time.time()
        print(format_route(a_star(cityStart, cityEnd)))
        end_time = time.time()
        print("%0.4f seconds" % (end_time-start_time))
