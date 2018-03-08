import numpy as np
import math  as math

class Vision:

    def __init__(self,dungeon,max_vision,model):
        self.max_vision = max_vision
        self.dungeon = dungeon
        self.width = len(self.dungeon)
        self.height = len(self.dungeon[0])
        self.D = {}
        self.model = model

        orientations = [8, 7, 4,  1,  2,  3,  6,  9]
        offset       = [180,225,270,315,0,45,90,135]


        for o in orientations:
            self.D[o] = {}
            for o1 in range(1,12):
                self.D[o][o1] = []

        for orient in range(len(orientations)):
            o = orientations[orient]
            ofs = offset[orient]

            for x in range(-max_vision,max_vision+1):
                for y in range(-max_vision,max_vision+1):
                    dist = np.sqrt(x**2+y**2)
                    if dist > max_vision or dist == 0:
                        continue
                    else:
                        a = self.angle(x,y,ofs)
                        dist = 1 - dist/max_vision
                        if   a > 180: continue
                        elif a <= 25: self.D[o][11].append((x,y,dist))
                        elif a <= 45: self.D[o][10].append((x,y,dist))
                        elif a <= 60: self.D[o][ 9].append((x,y,dist))
                        elif a <= 75: self.D[o][ 8].append((x,y,dist))
                        elif a <  90: self.D[o][ 7].append((x,y,dist))
                        elif a == 90: self.D[o][ 6].append((x,y,dist))
                        elif a < 105: self.D[o][ 5].append((x,y,dist))
                        elif a < 120: self.D[o][ 4].append((x,y,dist))
                        elif a < 145: self.D[o][ 3].append((x,y,dist))
                        elif a < 165: self.D[o][ 2].append((x,y,dist))
                        else:         self.D[o][ 1].append((x,y,dist))

        #fix overlap to prevent looking throught walls

        #SOUTH-WEST
        self.add( 1, 2,-1, 0)
        self.add( 1, 4,-1, 1)
        self.add( 1, 5,-1, 1)
        self.add( 1, 5,-2, 2)
        self.add( 1, 7,-1, 1)
        self.add( 1, 7,-2, 2)
        self.add( 1, 8,-1, 1)
        self.add( 1, 9, 0, 1)
        self.add( 1, 9, 0, 2)
        self.add( 1, 9, 0, 3)

        #SOUTH
        self.add( 2, 2,-1, 1)
        self.add( 2, 4,-1, 1)
        self.add( 2, 5, 0, 1)
        self.add( 2, 5, 0, 2)
        self.add( 2, 5,-1, 3)
        self.add( 2, 7, 0, 1)
        self.add( 2, 7, 0, 2)
        self.add( 2, 7, 1, 3)
        self.add( 2, 8, 1, 1)
        self.add( 2, 9, 1, 1)
        self.add( 2, 9, 2, 2)

        #SOUTH-EAST
        self.add( 3, 2, 0, 1)
        self.add( 3, 4, 1, 1)
        self.add( 3, 5, 1, 1)
        self.add( 3, 5, 2, 2)
        self.add( 3, 7, 1, 1)
        self.add( 3, 7, 2, 2)
        self.add( 3, 8, 1, 1)
        self.add( 3, 9, 1, 0)
        self.add( 3, 9, 2, 0)
        self.add( 3, 9, 3, 0)

        #WEST
        self.add( 4, 2,-1,-1)
        self.add( 4, 4,-1,-1)
        self.add( 4, 5,-1, 0)
        self.add( 4, 5,-2, 0)
        self.add( 4, 5,-3,-1)
        self.add( 4, 7,-1, 0)
        self.add( 4, 7,-2, 0)
        self.add( 4, 7,-3, 1)
        self.add( 4, 8,-1, 1)
        self.add( 4, 9,-1, 1)
        self.add( 4, 9,-2, 2)

        #EAST
        self.add( 6, 2, 1, 1)
        self.add( 6, 4, 1, 1)
        self.add( 6, 5, 1, 0)
        self.add( 6, 5, 2, 0)
        self.add( 6, 5, 3, 1)
        self.add( 6, 7, 1, 0)
        self.add( 6, 7, 2, 0)
        self.add( 6, 7, 3,-1)
        self.add( 6, 8, 1,-1)
        self.add( 6, 9, 1,-1)
        self.add( 6, 9, 2,-2)

        #NORTH-WEST
        self.add( 7, 2, 0,-1)
        self.add( 7, 4,-1,-1)
        self.add( 7, 5,-1,-1)
        self.add( 7, 5,-2,-2)
        self.add( 7, 7,-1,-1)
        self.add( 7, 7,-2,-2)
        self.add( 7, 8,-1,-1)
        self.add( 7, 9,-1, 0)
        self.add( 7, 9,-2, 0)
        self.add( 7, 9,-3, 0)
        
        #NORTH
        self.add( 8, 2, 1,-1)
        self.add( 8, 4, 1,-1)
        self.add( 8, 5, 0,-1)
        self.add( 8, 5, 0,-2)
        self.add( 8, 5, 1,-3)
        self.add( 8, 7, 0,-1)
        self.add( 8, 7, 0,-2)
        self.add( 8, 7,-1,-3)
        self.add( 8, 8,-1,-1)
        self.add( 8, 9,-1,-1)
        self.add( 8, 9,-2,-2)
        
        #NORTH-EAST
        self.add( 9, 2, 1, 0)
        self.add( 9, 4, 1,-1)
        self.add( 9, 5, 1,-1)
        self.add( 9, 5, 2,-2)
        self.add( 9, 7, 1,-1)
        self.add( 9, 7, 2,-2)
        self.add( 9, 8, 1,-1)
        self.add( 9, 9, 0,-1)
        self.add( 9, 9, 0,-2)
        self.add( 9, 9, 0,-3)
        
        for o in self.D:
            for o1 in self.D[o]:
                self.D[o][o1].sort(key=lambda coord: coord[2], reverse=True )

    def add(self,orientation,direction,x,y):
        self.D[orientation][direction].append((x,y,self.dist(x,y)))

    def dist(self,x,y):
        return 1 - np.sqrt(x**2+y**2)/self.max_vision
                        
    
    def angle(self,x,y,ofs):
        a = (90 if y >= 0 else 270) if x == 0 else 180/math.pi * np.arctan(y/x)
        a = (a+ofs)%360
        if x < 0:   a = (a+180)%360
        elif y < 0: a = (a+360)%360
        return a
        
    def getVision(self, x, y, orientation,field):
        for coords in self.D[orientation][field]:
            dx = x + coords[0]
            dy = y + coords[1]
            if dx < 0 or dx >= self.width or dy < 0 or dy >= self.height:
                return (0,0)
            if not self.dungeon[ dx ][ dy ].isEmpty():
                e_name = self.dungeon[dx][dy].getFirstElement()
                if e_name == 'wall':
                    return (coords[2],0)
                else:
                    #print( self.model.getEntities()[e_name] )
                    return (coords[2],self.model.getEntities()[e_name].getCHeat())
        return (0,0)

if __name__ == '__main__':
    V = Vision(np.zeros((25,20)),11)
