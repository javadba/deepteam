import numpy as np
from .planner import load_model
from .detector import load_det_model
import torchvision.transforms.functional as TF
import torchvision.transforms as this_is_annoying
from os import path
import PIL
import torch

import torch.utils.tensorboard as tb
train_logger = tb.SummaryWriter(path.join("logs_1", 'train_3'))

velocity_list = []

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


    def __init__(self, player_id=0):
        """
        Set up a soccer player.
        The player_id starts at 0 and increases by one for each player added. You can use the player id to figure out your team (player_id % 2), or assign different roles to different agents.
        """
        all_players = ['adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley', 'kiki', 'konqi', 'nolok',
                       'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux', 'wilber', 'xue']
        self.kart = all_players[np.random.choice(len(all_players))]
        self.player_id = player_id

    def act(self, image, player_info):
        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart.
        return: Dict describing the action
        """
        #LOCATIONS ARE ALWAYS STORED AS [X,H,Y] WHERE X IS WIDTH OF MAP, H IS HEIGHT OFF THE GROUND, AND Y IS NORTH-SOUTH ON THE MAP
        red_net = [0, 0.07000000029802322, -64.5]
        blue_net = [0, 0.07000000029802322, 64.5]
        field_midpoint =[0,0,0]

        far_left = [-45, 0, 0]
        far_right = [45, 0, 0]

        #CONVERTS A 3D LOCATION TO AN X-Y GIVEN THE CAMERA AND PROJECTION OF ANY GIVEN PLAYER
        def _to_image(x, proj, view):
            p = proj @ view @ np.array(list(x) + [1])
            return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

        if (self.player_id == 0):

            #IMPORTANT 3D LOCATIONS
            home_net_location = red_net
            target_net_location = blue_net
            kart_location = player_info.kart.location
            frame_number = len(velocity_list)
            print("Frame Number: ", frame_number)
            train_logger.add_scalar('rotation_1', player_info.kart.rotation[1], global_step = frame_number)
            train_logger.add_scalar('rotation_2', player_info.kart.rotation[3], global_step = frame_number)

            #ESTABLISH CAMERA VIEWS
            proj = np.array(player_info.camera.projection).T
            view = np.array(player_info.camera.view).T

            #GENERATE X-Y FOR HOME NET, TARGET NET, AND HOCKEY PUCK

            #HOME NET
            home_point_image = _to_image(home_net_location, proj, view)
            home_x_location = home_point_image[0]
            home_y_location = home_point_image[1]

            #TARGET NET
            target_point_image = _to_image(target_net_location, proj, view)
            target_x_location = target_point_image[0]
            target_y_location = target_point_image[1]

            #LOAD TRAINED MODEL
            planner = load_model()
            detector = load_det_model()

            #PUCK

            #IN_VIEW?
            view_conf = detector(TF.to_tensor(image)[None]).squeeze(0).cpu().detach().numpy())
            #LOC
            puck_point_image = planner(TF.to_tensor(image)[None]).squeeze(0).cpu().detach().numpy()
            puck_x_location = puck_point_image[0]
            puck_y_location = puck_point_image[1]
            #puck_visible = puck_point_image[2]
            print(puck_point_image)

            #CONTROLLER USING X-Y POINTS
            current_vel = abs(max(player_info.kart.velocity))
            velocity_list.append(current_vel)
            

            #SET ACTION DEFAULTS
            nitro = False
            rescue = False
            brake = False
            drift= False
            acceleration = .1
            steer = 1
            buffer = 0

            #BUILD OUT LOGIC FOR DECISIONS
            #im = Image.fromarray((img[0] * 255).astype(np.uint8))
            
            im = PIL.Image.fromarray(image,'RGB')
            transform =  this_is_annoying.ToTensor()
            if frame_number % 50 == 0:
              train_logger.add_image('viz', transform(im), frame_number)
          
      
            #ACTIONS ARE SAVED HERE
            action = {'acceleration': acceleration, 'brake': brake, 'drift': drift, 'nitro': nitro, 'rescue': rescue, 'steer': steer}

        if (self.player_id == 1):
            target_net_location = red_net

            def _to_image(x, proj, view):
                p = proj @ view @ np.array(list(x) + [1])
                return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

            proj = np.array(player_info.camera.projection).T
            view = np.array(player_info.camera.view).T
            aim_point_world = target_net_location
            aim_point_image = _to_image(aim_point_world, proj, view)
            print(aim_point_image)

            current_vel = abs(max(player_info.kart.velocity))
            print(current_vel)

            # this the point driver
            aim_x_location = aim_point_image[0]
            aim_y_location = aim_point_image[1]

            brake = False

            acceleration = 1 * (1 - abs(aim_x_location))

            if aim_x_location < 0:
                steer = -1
            elif aim_x_location > 0:
                steer = 1
            else:
                steer = 0

            if aim_x_location < -0.4:
                drift = True
            elif aim_x_location > 0.4:
                drift = True
            else:
                drift = False

            if abs(aim_y_location) > .3:
                acceleration = 0
                brake = True

            nitro = False
            rescue = False

            action = {'acceleration': acceleration, 'brake': brake, 'drift': drift, 'nitro': nitro, 'rescue': rescue,
                      'steer': steer}

        if (self.player_id == 2):
            target_net_location = blue_net

            def _to_image(x, proj, view):
                p = proj @ view @ np.array(list(x) + [1])
                return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

            proj = np.array(player_info.camera.projection).T
            view = np.array(player_info.camera.view).T
            aim_point_world = target_net_location
            aim_point_image = _to_image(aim_point_world, proj, view)
            print(aim_point_image)

            current_vel = abs(max(player_info.kart.velocity))
            print(current_vel)

            # this the point driver
            aim_x_location = aim_point_image[0]
            aim_y_location = aim_point_image[1]

            brake = False

            acceleration = 1 * (1 - abs(aim_x_location))

            if aim_x_location < 0:
                steer = -1
            elif aim_x_location > 0:
                steer = 1
            else:
                steer = 0

            if aim_x_location < -0.4:
                drift = True
            elif aim_x_location > 0.4:
                drift = True
            else:
                drift = False

            if abs(aim_y_location) > .3:
                acceleration = 0
                brake = True

            nitro = False
            rescue = False

            action = {'acceleration': acceleration, 'brake': brake, 'drift': drift, 'nitro': nitro, 'rescue': rescue,
                      'steer': steer}

        if (self.player_id == 3):
            target_net_location = red_net

            def _to_image(x, proj, view):
                p = proj @ view @ np.array(list(x) + [1])
                return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

            proj = np.array(player_info.camera.projection).T
            view = np.array(player_info.camera.view).T
            aim_point_world = target_net_location
            aim_point_image = _to_image(aim_point_world, proj, view)
            print(aim_point_image)

            current_vel = abs(max(player_info.kart.velocity))
            print(current_vel)

            # this the point driver
            aim_x_location = aim_point_image[0]
            aim_y_location = aim_point_image[1]

            brake = False

            acceleration = 1 * (1 - abs(aim_x_location))

            if aim_x_location < 0:
                steer = -1
            elif aim_x_location > 0:
                steer = 1
            else:
                steer = 0

            if aim_x_location < -0.4:
                drift = True
            elif aim_x_location > 0.4:
                drift = True
            else:
                drift = False

            if abs(aim_y_location) > .3:
                acceleration = 0
                brake = True

            nitro = False
            rescue = False

            action = {'acceleration': acceleration, 'brake': brake, 'drift': drift, 'nitro': nitro, 'rescue': rescue,
                      'steer': steer}

        return action

