import numpy as np
from .planner import load_model
import torchvision.transforms.functional as TF
import math

location_list_x = []
location_list_y = []
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

    #Notes
    # train with a single player
    # If low confidence for puck location, spin around until confidence is high
    # rescue always points at the red net
    # line up puck and net
    # rotation to always point to the goal
    #   Two different points in 4D can be the same location in 3D
    # If loss isnt good, increase model size



    def __init__(self, player_id=0):
        """
        Set up a soccer player.
        The player_id starts at 0 and increases by one for each player added. You can use the player id to figure out your team (player_id % 2), or assign different roles to different agents.
        """
        all_players = ['adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley', 'kiki', 'konqi', 'nolok',
                       'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux', 'wilber', 'xue']

        self.kart = 'tux'
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
            player_o_location = player_info.kart.location

            #IMPORTANT 3D LOCATIONS
            home_net_location = red_net
            target_net_location = blue_net
            kart_location = player_info.kart.location

            #Tracking location through time
            step = len(location_list_x)
            location_list_x.append(kart_location[0])
            location_list_y.append(kart_location[2])

            #if (abs(location_list_x[step]) + abs(location_list_y[step])) > location_list_x[step-1]:
            #    print(location_list_x[step])
            #    print(location_list_x[step-1])
            #    print("MOVING ")
            #else:
            #    print("STUCK ")

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

            #PUCK
            puck_point_image = planner(TF.to_tensor(image)[None]).squeeze(0).cpu().detach().numpy()
            puck_x_location = puck_point_image[0]
            puck_y_location = puck_point_image[1]

            #CONTROLLER USING X-Y POINTS
            current_vel = abs(max(player_info.kart.velocity))
            velocity_list.append(current_vel)

            #SET ACTION DEFAULTS
            nitro = True
            rescue = False
            brake = False
            drift = False
            steer = 1
            buffer = 1
            acceleration = 0.1

            def convert_to_angles(list):
                w = list[0]
                x = list[1]
                y = list[2]
                z = list[3]

                angles = []

                # roll(x - axis rotation)
                sinr_cosp = 2 * (w * x + y * z)
                cosr_cosp = 1 - 2 * (x * x + y * y)
                roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))
                angles.append(roll)

                #pitch(y - axis rotation)

                sinp = 2 * (w * y - z * x)

                if (abs(sinp) >= 1):
                    pitch = math.degrees(math.copysign(math.pi / 2, sinp)) #use 90degrees if outofrange
                else:
                    pitch = math.degrees(math.asin(sinp))
                angles.append(pitch)


                #Yee Yaw
                siny_cosp = 2 * (w * z + x * y)
                cosy_cosp = 1 - 2 * (y * y + z * z)
                yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
                angles.append(yaw)

                return angles

            #print(player_info.kart.rotation)

            # from Red Team's perspective
            if abs(convert_to_angles(player_info.kart.rotation)[0]) > 170:
                north = 0
            else:
                north = 1
            # left is 90, right is -90
            east_west = convert_to_angles(player_info.kart.rotation)[1]

            print(north, east_west)

            #print(convert_to_angles(player_info.kart.rotation))
            print(step)


            #BUILD OUT LOGIC FOR DECISIONS
            #if puck_x_location < 0:
            #    steer = -1*buffer
            #elif puck_x_location > 0:
            #    steer = 1*buffer
            #else:
            #    steer = 0

            #if puck_y_location > 0.2:
            #    acceleration = 0.0
            #    brake = True

            #if current_vel > 15:
            #    acceleration = 0
            #    brake = True

            #RESCUE FUNCTION
            #print(player_info.kart.location)

            #ACTIONS ARE SAVED HERE
            action = {'acceleration': acceleration, 'brake': brake, 'drift': drift, 'nitro': nitro, 'rescue': rescue, 'steer': steer}

        if (self.player_id == 1):
            # IMPORTANT 3D LOCATIONS
            home_net_location = red_net
            target_net_location = blue_net
            kart_location = player_info.kart.location

            # ESTABLISH CAMERA VIEWS
            proj = np.array(player_info.camera.projection).T
            view = np.array(player_info.camera.view).T

            # GENERATE X-Y FOR HOME NET, TARGET NET, AND HOCKEY PUCK

            # HOME NET
            home_point_image = _to_image(home_net_location, proj, view)
            home_x_location = home_point_image[0]
            home_y_location = home_point_image[1]

            # TARGET NET
            target_point_image = _to_image(target_net_location, proj, view)
            target_x_location = target_point_image[0]
            target_y_location = target_point_image[1]

            # LOAD TRAINED MODEL
            planner = load_model()

            # PUCK
            puck_point_image = planner(TF.to_tensor(image)[None]).squeeze(0).cpu().detach().numpy()
            puck_x_location = puck_point_image[0]
            puck_y_location = puck_point_image[1]
            print(player_info.kart.rotation)

            # CONTROLLER USING X-Y POINTS
            current_vel = abs(max(player_info.kart.velocity))

            # SET ACTION DEFAULTS
            nitro = False
            rescue = False
            brake = False
            drift = False
            acceleration = 1 #* (1 - abs(puck_x_location))
            steer = 0
            buffer = 0.5

            # BUILD OUT LOGIC FOR DECISIONS
            #if puck_x_location < 0:
            #    steer = -1 * buffer
            #elif puck_x_location > 0:
            #    steer = 1 * buffer
           # else:
            #    steer = 0

            #if puck_y_location > 0.2 or abs(puck_x_location) > 0.8:
            #    acceleration = 0.0
            #    brake = True

            # ACTIONS ARE SAVED HERE
            action = {'acceleration': acceleration, 'brake': brake, 'drift': drift, 'nitro': nitro, 'rescue': rescue,
                      'steer': steer}

        if (self.player_id == 2):
            # IMPORTANT 3D LOCATIONS
            home_net_location = red_net
            target_net_location = blue_net
            kart_location = player_info.kart.location

            # ESTABLISH CAMERA VIEWS
            proj = np.array(player_info.camera.projection).T
            view = np.array(player_info.camera.view).T

            # GENERATE X-Y FOR HOME NET, TARGET NET, AND HOCKEY PUCK

            # HOME NET
            home_point_image = _to_image(home_net_location, proj, view)
            home_x_location = home_point_image[0]
            home_y_location = home_point_image[1]

            # TARGET NET
            target_point_image = _to_image(target_net_location, proj, view)
            target_x_location = target_point_image[0]
            target_y_location = target_point_image[1]

            # LOAD TRAINED MODEL
            planner = load_model()

            # PUCK
            puck_point_image = planner(TF.to_tensor(image)[None]).squeeze(0).cpu().detach().numpy()
            puck_x_location = puck_point_image[0]
            puck_y_location = puck_point_image[1]
            print(puck_point_image)

            # CONTROLLER USING X-Y POINTS
            current_vel = abs(max(player_info.kart.velocity))

            # SET ACTION DEFAULTS
            nitro = False
            rescue = False
            brake = False
            drift = False
            acceleration = 1 * (1 - abs(puck_x_location))
            steer = 0
            buffer = 0.5

            # BUILD OUT LOGIC FOR DECISIONS
            if puck_x_location < 0:
                steer = -1 * buffer
            elif puck_x_location > 0:
                steer = 1 * buffer
            else:
                steer = 0

            if puck_y_location > 0.2 or abs(puck_x_location) > 0.8:
                acceleration = 0.0
                brake = True

            # ACTIONS ARE SAVED HERE
            action = {'acceleration': acceleration, 'brake': brake, 'drift': drift, 'nitro': nitro, 'rescue': rescue,
                      'steer': steer}

        if (self.player_id == 3):
            # IMPORTANT 3D LOCATIONS
            home_net_location = red_net
            target_net_location = blue_net
            kart_location = player_info.kart.location

            # ESTABLISH CAMERA VIEWS
            proj = np.array(player_info.camera.projection).T
            view = np.array(player_info.camera.view).T

            # GENERATE X-Y FOR HOME NET, TARGET NET, AND HOCKEY PUCK

            # HOME NET
            home_point_image = _to_image(home_net_location, proj, view)
            home_x_location = home_point_image[0]
            home_y_location = home_point_image[1]

            # TARGET NET
            target_point_image = _to_image(target_net_location, proj, view)
            target_x_location = target_point_image[0]
            target_y_location = target_point_image[1]

            # LOAD TRAINED MODEL
            planner = load_model()

            # PUCK
            puck_point_image = planner(TF.to_tensor(image)[None]).squeeze(0).cpu().detach().numpy()
            puck_x_location = puck_point_image[0]
            puck_y_location = puck_point_image[1]
            print(puck_point_image)

            # CONTROLLER USING X-Y POINTS
            current_vel = abs(max(player_info.kart.velocity))

            # SET ACTION DEFAULTS
            nitro = False
            rescue = False
            brake = False
            drift = False
            acceleration = 1 * (1 - abs(puck_x_location))
            steer = 0
            buffer = 0.5

            # BUILD OUT LOGIC FOR DECISIONS
            if puck_x_location < 0:
                steer = -1 * buffer
            elif puck_x_location > 0:
                steer = 1 * buffer
            else:
                steer = 0

            if puck_y_location > 0.2 or abs(puck_x_location) > 0.8:
                acceleration = 0.0
                brake = True

            # ACTIONS ARE SAVED HERE
            action = {'acceleration': acceleration, 'brake': brake, 'drift': drift, 'nitro': nitro, 'rescue': rescue,
                      'steer': steer}

        return action

