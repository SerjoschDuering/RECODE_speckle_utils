def gh_color_blueRed():
    # grasshoper color scheme 
    color_list = [[15,16,115],
            [177,198,242],
            [251,244,121],
            [222,140,61],
            [183,60,34]]
    # Scale RGB values to [0,1] range
    color_list = [[c/255. for c in color] for color in color_list]
    return color_list

def gh_color_whiteRed():
    # grasshoper color scheme 
    color_list = [[255,255,255],
            [111,19,12],
            ]
    # Scale RGB values to [0,1] range
    color_list = [[c/255. for c in color] for color in color_list]
    return color_list

#---
#---