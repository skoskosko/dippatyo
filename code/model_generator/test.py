def dfs(grid, visited, row, col, group):
    # Define the directions to explore (up, down, left, right, diagonals)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    # Mark the current cell as visited
    visited[row][col] = True
    group.append((row, col))
    
    # Explore adjacent cells
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < len(grid) and 0 <= new_col < len(grid[0]) and grid[new_row][new_col] == 1 and not visited[new_row][new_col]:
            dfs(grid, visited, new_row, new_col, group)

def group_coordinates(grid):
    visited = [[False for _ in range(len(grid[0]))] for _ in range(len(grid))]
    groups = []
    
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1 and not visited[i][j]:
                group = []
                dfs(grid, visited, i, j, group)
                groups.append(group)
                
    return groups

# Example usage:
coordinates = [
    [1, 0, 1],
    [1, 0, 0],
    [1, 0, 1]
]

groups = group_coordinates(coordinates)
print(groups)