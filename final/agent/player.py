import numpy as np


class HockeyPlayer:
    """
       Your ice hockey player. You may do whatever you want here. There are three rules:
        1. no calls to the pystk library (your code will not run on the tournament system if you do)
        2. There needs to be a deep network somewhere in the loop
        3. You code must run in 100 ms / frame on a standard desktop CPU (no for testing GPU)
        
        Try to minimize library dependencies, nothing that does not install through pip on linux.
    """
    
    """
       You may request to play with a different kart.
       Call `python3 -c "import pystk; pystk.init(pystk.GraphicsConfig.ld()); print(pystk.list_karts())"` to see all values.
    """
    kart = ""

    # dont need to identify the net - the net is always at a fixed abolute location - tricky thing is need to identify what team you are
    # if team == blue
    #       net = [10]
        #else:
            #net = [-10]

    # We have access to our x,y,z location at all times - verify on piazza

    # In theory, we can share our teammates location via communication
            # we know everyhting except where the puck is and where the opponnents

    #Try a lot of things and include in the report

    #two big issues:
    #       1) if you are really far from the puck, it is really small
            #2) what to do if the puck is outside the image?

    # AI will cream you

    blue_net = [[-10.449999809265137, 0.07000000029802322, -64.5], [10.449999809265137, 0.07000000029802322, -64.5]]
    red_net = [[10.460000038146973, 0.07000000029802322, 64.5], [-10.510000228881836, 0.07000000029802322, 64.5]]

    def __init__(self, player_id = 0):
        """
        Set up a soccer player.
        The player_id starts at 0 and increases by one for each player added. You can use the player id to figure out your team (player_id % 2), or assign different roles to different agents.
        """
        
    def act(self, image, player_info):
        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart.
        return: Dict describing the action
        """
        action = {'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': np.random.uniform(-1, 1)}
        """
        Your code here.
        """

        return action

