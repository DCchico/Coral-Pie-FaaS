import csv

# Function to load the camera connections from the CSV file
def load_camera_connections(filename):
    camera_connections = {}
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            parts = row[0].split(': ')
            camera_name = parts[0]
            reachable_cameras = parts[1].split(', ')
            camera_connections[camera_name] = reachable_cameras
    return camera_connections

# Function to find reachable cameras for a given camera name
def find_reachable_cameras(camera_connections, camera_name):
    if camera_name in camera_connections:
        return camera_connections[camera_name]
    else:
        return []