import numpy as np

channelmap = {
    'Xrotation' : 'x',
    'Yrotation' : 'y',
    'Zrotation' : 'z'   
}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x' : 0,
    'y' : 1,
    'z' : 2,
}
def save(filename, rotations, vpositions, jointoffset, parentslist, names=None, frametime=1.0/24.0, order='zyx', positions=False, orients=True):
    """
    Saves an Animation to file as BVH
    
    Parameters
    ----------
    filename: str
        File to be saved to
        
    anim : Animation
        Animation to save
        
    names : [str]
        List of joint names
    
    order : str
        Optional Specifier for joint order.
        Given as string E.G 'xyz', 'zxy'
    
    frametime : float
        Optional Animation Frame time
        
    positions : bool
        Optional specfier to save bone
        positions for each frame
        
    orients : bool
        Multiply joint orients to the rotations
        before saving.
        
    """
    
    if names is None:
        names = ["joint_" + str(i) for i in range(len(parentslist))]
    
    with open(filename, 'w') as f:

        t = ""
        f.write("%sHIERARCHY\n" % t)
        f.write("%sROOT %s\n" % (t, names[0]))
        f.write("%s{\n" % t)
        t += '\t'

        f.write("%sOFFSET %f %f %f\n" % (t, jointoffset[0][0][0], jointoffset[0][0][1], jointoffset[0][0][2]) )
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % 
            (t, channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))

        for i in range(22):
            if parentslist[i] == -1:
                t = save_joint(f, jointoffset, parentslist, names, t, i, order=order, positions=positions)

        t = t[:-1]
        f.write("%s}\n" % t)

        f.write("MOTION\n")
        f.write("Frames: %i\n" % len(rotations))
        f.write("Frame Time: %f\n" % frametime)
            
        #if orients:        
        #    rots = np.degrees((-anim.orients[np.newaxis] * anim.rotations).euler(order=order[::-1]))
        #else:
        #    rots = np.degrees(anim.rotations.euler(order=order[::-1]))
        rots = np.degrees(rotations)
        poss = vpositions
        
        for i in range(len(rotations)):
            for j in range(len(jointoffset)):
                
                if vpositions or j == 0:
                
                    f.write("%f %f %f %f %f %f " % (
                        poss[i][j][0],                  poss[i][j][1],                  poss[i][j][2], 
                        rots[i][j][ordermap[order[0]]], rots[i][j][ordermap[order[1]]], rots[i][j][ordermap[order[2]]]))
                
                else:
                    
                    f.write("%f %f %f " % (
                        rots[i][j][ordermap[order[0]]], rots[i][j][ordermap[order[1]]], rots[i][j][ordermap[order[2]]]))

            f.write("\n")
    
    
def save_joint(f, jointoffset, parentlist, names, t, i, order='zyx', positions=False):
    
    f.write("%sJOINT %s\n" % (t, names[i]))
    f.write("%s{\n" % t)
    t += '\t'
  
    f.write("%sOFFSET %f %f %f\n" % (t, jointoffset[0][0][0], jointoffset[0][0][1], jointoffset[0][0][2]) )
    
    if positions:
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % (t, 
            channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))
    else:
        f.write("%sCHANNELS 3 %s %s %s\n" % (t, 
            channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))
    
    end_site = True
    
    for j in range(22):
        if parentlist[j] == i:
            t = save_joint(f, jointoffset, parentlist, names, t, j, order=order, positions=positions)
            end_site = False
    
    if end_site:
        f.write("%sEnd Site\n" % t)
        f.write("%s{\n" % t)
        t += '\t'
        f.write("%sOFFSET %f %f %f\n" % (t, 0.0, 0.0, 0.0))
        t = t[:-1]
        f.write("%s}\n" % t)
  
    t = t[:-1]
    f.write("%s}\n" % t)
    
    return t