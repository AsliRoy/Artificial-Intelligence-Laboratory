#Define graph for performing BFS and DFS operation
graph = {'A': ['B', 'C','E'],
         'B': ['A', 'D', 'E'],
         'C': ['A', 'F','G'],
         'D': ['B','E'],
         'E': ['A','B', 'D'],
         'F': ['C'],
         'G': ['C'] }

#Define BFS function
def bfs(graph, start):
    #Initialize traversed and queue lists
    traversed = []
    queue = [start]
    #Pop condition for queue
    while queue:
        vertex = queue.pop(0)
        if vertex not in traversed:
            traversed.append(vertex)
            neighbours = graph[vertex]
            
            #Append neighbours
            for neighbour in neighbours:
                queue.append(neighbour)
    #Return traversal order            
    return traversed

#Driver function call
print ("Breadth first search of the given graph is: ")
print( bfs(graph, 'A')) 

#Define DFS function
def dfs(graph, start):
    #Initialize traversed and stack lists
    traversed = []
    stack = [start]
    #Pop condition for stack
    while stack:
        vertex = stack.pop()
        if vertex not in traversed:
            traversed.append(vertex)
            neighbours = graph[vertex]
            
            #Append neighbours
            for neighbour in neighbours:
                stack.append(neighbour)
                
    #Return traversal order
    return traversed

#Driver function call
print ("Depth first search of the given graph is : " )
print ( dfs(graph, 'A')) 



