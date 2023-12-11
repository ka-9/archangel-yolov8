import heapq
import random
import serial

#---------------------------------------------------------------------------------------------------------------------------------
# bluetooth port setup
# Find the Bluetooth COM port dynamically
def find_com_port():
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    for port, _, _ in ports:
        if "Bluetooth" in port:
            return port
    return None

bluetooth_port = find_com_port()
ser = serial.Serial(bluetooth_port, 9600, timeout=1)

#---------------------------------------------------------------------------------------------------------------------------------
# to send with bluetooth

def send_array_with_size(data_array):
    array_size = len(data_array)
    data_string = f"{array_size}," + ','.join(map(str, data_array)) + "\n"
 
    #ser.write(data_string.encode('utf-8'))
    #time.sleep(1)  # Allow time for data to be sent


#---------------------------------------------------------------------------------------------------------------------------------
# path finding algorithm    
    
def dijkstra_maze(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Possible movement directions

    # Initialize distances to all cells as infinity except for the start cell.
    distances = [[float('inf')] * cols for _ in range(rows)]
    distances[start[0]][start[1]] = 0
    
    # Initialize a dictionary to track the previous cell in the path.
    previous = {start: None}

    # Priority queue to keep track of cells with their current distance.
    priority_queue = [(0, start)]  # Format: (distance, (row, col))

    while priority_queue:
        current_distance, (row, col) = heapq.heappop(priority_queue)
        
        # Ignore this cell if we've found a shorter path to it.
        if current_distance > distances[row][col]:
            continue
        
        # Explore neighboring cells
        for dr, dc in directions:
            r, c = row + dr, col + dc

            # Check if the neighboring cell is within the maze boundaries
            if 0 <= r < rows and 0 <= c < cols:
                new_distance = current_distance + 1  # In this version, all cells have a weight of 1

                # If a shorter path is found to the neighboring cell, update the distance and previous cell.
                if new_distance < distances[r][c] and maze[r][c] == 0:
                    distances[r][c] = new_distance
                    previous[(r, c)] = (row, col)
                    heapq.heappush(priority_queue, (new_distance, (r, c)))

    # Check if the end cell is still unreachable (distance remains infinite).
    if distances[end[0]][end[1]] == float('inf'):
        return "Maze is unsolvable", None, None

    # Reconstruct the path from the end cell to the start cell.
    path = []
    current_cell = end
    while current_cell:
        path.append(current_cell)
        current_cell = previous[current_cell]
    path.reverse()  # Reverse the path to start from the beginning.

    # Translate the list of cells into text directions.
    def translate_to_text(path):
        directions = []
        for i in range(len(path) - 1):
            current_row, current_col = path[i]
            next_row, next_col = path[i + 1]

            if current_row < next_row:
                directions.append("down")
            elif current_row > next_row:
                directions.append("up")
            elif current_col < next_col:
                directions.append("right")
            elif current_col > next_col:
                directions.append("left")
        return directions
    
    text_directions = translate_to_text(path)
    
    # Return the shortest distance, the path, and text directions.
    return distances[end[0]][end[1]], path, text_directions

# Example usage:
# Define a maze as a 2D matrix with 0 for open areas and 1 for obstacles.

#---------------------------------------------------------------------------------------------------------------------------------
# maze example

def generate_random_maze(rows, cols, density):
    # Generate a random maze using a given density (0.0 to 1.0).
    maze = [[0 if random.random() > density else 1 for _ in range(cols)] for _ in range(rows)]
    maze[0][0] = 0  # Start cell
    maze[rows - 1][cols - 1] = 0  # End cell
    return maze

def print_maze(maze):
    for row in maze:
        print(" ".join([str(cell) for cell in row]))

# Example usage:
rows = 10
cols = 10
density = 0.4  # Adjust this value to control maze complexity



maze = generate_random_maze(rows, cols, density)
print_maze(maze)

start_cell = (0, 0)
end_cell = (rows-1, cols-1)

# basically the matrix output will go instead of "maze", the rover position instead of "start_cell", and the person location instead of
# "end_cell"
result, path, text_directions = dijkstra_maze(maze, start_cell, end_cell)




if result == "Maze is unsolvable":
    print(result)
else:
    send_array_with_size(text_directions)
    print("Shortest distance from {start_cell} to {end_cell}: {result}")
    print("Text Directions:")
    for direction in text_directions:
        print(direction)