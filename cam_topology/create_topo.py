import csv

# Define the dimensions of the grid
rows = 4
cols = 4

# Initialize an empty grid with placeholders
grid = [[None] * cols for _ in range(rows)]

# Fill in the reachable cameras for each camera
for i in range(rows):
    for j in range(cols):
        camera_name = f'camera_{i * cols + j + 1}'
        reachable_cameras = []
        
        if i > 0:
            reachable_cameras.append(f'camera_{(i-1) * cols + j + 1}')
        if i < rows - 1:
            reachable_cameras.append(f'camera_{(i+1) * cols + j + 1}')
        if j > 0:
            reachable_cameras.append(f'camera_{i * cols + j}')
        if j < cols - 1:
            reachable_cameras.append(f'camera_{i * cols + j + 2}')
        
        grid[i][j] = f'{camera_name}: ' + ', '.join(reachable_cameras)

# Create and write the grid data to a CSV file
with open('camera_connections.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    for row in grid:
        for entry in row:
            csvwriter.writerow([entry])

