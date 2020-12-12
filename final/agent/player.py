import numpy as np
from .planner import load_model
from .detector import load_det_model
import torchvision.transforms.functional as TF
import math
import statistics
import torch.nn.functional as F
import torch
from scipy import stats

#SHARED GLOBAL VARIABLES
# LOCATIONS ARE ALWAYS STORED AS [X,H,Y] WHERE X IS WIDTH OF MAP, H IS HEIGHT OFF THE GROUND, AND Y IS NORTH-SOUTH ON THE MAP

red_net = [0, 0.07000000029802322, -64.5]
blue_net = [0, 0.07000000029802322, 64.5]

field_midpoint =[0,0,0]

far_left = [-45, 0, 0]
far_right = [45, 0, 0]

#SHARED GLOBAL FUNCTIONS

    #CONVERTS A 3D LOCATION TO AN X-Y GIVEN THE CAMERA AND PROJECTION OF ANY GIVEN PLAYER
def _to_image(x, proj, view):
    p = proj @ view @ np.array(list(x) + [1])
    return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

    #CONVERTS A PLAYER'S ROTATION FROM A QUARTERNION TO A USABLE ANGLE
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

    # pitch(y - axis rotation)
    sinp = 2 * (w * y - z * x)

    if (abs(sinp) >= 1):
        pitch = math.degrees(math.copysign(math.pi / 2, sinp))  # use 90degrees if outofrange
    else:
        pitch = math.degrees(math.asin(sinp))
    angles.append(pitch)

    # Yaw
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
    angles.append(yaw)

    return angles

#PLAYER1 GLOBAL VARIABLES
player_one_location_x = [0]
player_one_location_y = [0]
player_one_velocity = []
player_one_volatility = []
player_one_puck_x_average = [0]
detector_tracker = [1]

#PLAYER2 GLOBAL VARIABLES
player_two_location_x = [0]
player_two_location_y = [0]
player_two_velocity = []
player_two_volatility = []
player_two_puck_x_average = [0]

#PLAYER3 GLOBAL VARIABLES
player_three_location_x = [0]
player_three_location_y = [0]
player_three_velocity = []
player_three_volatility = []
player_three_puck_x_average = [0]

#PLAYER4 GLOBAL VARIABLES
player_four_location_x = [0]
player_four_location_y = [0]
player_four_velocity = []
player_four_volatility = []
player_four_puck_x_average = [0]


class HockeyPlayer:
    kart = ""
    def __init__(self, player_id=0):
        self.kart = 'tux'
        self.player_id = player_id

    def act(self, image, player_info):

        if (self.player_id == 0):

            #IMPORTANT 3D LOCATIONS
            home_net_location = red_net
            target_net_location = blue_net
            player_one_location = player_info.kart.location

            #PLAYER TEMPORAL TRACKING
            step = len(player_one_location_x)
            player_one_location_x.append(player_one_location[0])
            player_one_location_y.append(player_one_location[2])
            current_vel = abs(max(player_info.kart.velocity))
            player_one_velocity.append(current_vel)

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
            puck_point_image = planner(TF.to_tensor(image)[None]).squeeze(0).cpu().detach().numpy()
            puck_visibility = detector(TF.to_tensor(image)[None]).squeeze(0).cpu().detach().numpy()

            puck_x_location = puck_point_image[0]
            puck_y_location = puck_point_image[1]
            puck_z_location = puck_visibility
            detector = np.argmax(puck_z_location)
            detector_tracker.append(detector)
            #print("\n", detector)
            #print(puck_z_location[detector])

            player_one_puck_x_average.append(puck_x_location)

            vol_check = []
            for i in player_one_puck_x_average[-10:]:
                vol_check.append(float(i))

            #CONTROLLER USING X-Y POINTS
            #SET ACTION DEFAULTS
            nitro = True
            rescue = False
            brake = False
            drift = False
            steer = 0
            acceleration = 0.2

            #DETERMINE NORTH SOUTH KART ROTATION
            if abs(convert_to_angles(player_info.kart.rotation)[0]) > 170:
                north = 0
            else:
                north = 1

            #DETERMINE EAST WEST KART ROATION (left is 90, right is -90)
            east_west = convert_to_angles(player_info.kart.rotation)[1]

            #BUILD OUT LOGIC FOR DECISIONS
            #LAUNCH CONTROL
            if step <= 43:
                print("Initial Launch!")
                if puck_x_location < 0:
                    steer = -0.7
                    acceleration = 0.95
                elif puck_x_location > 0:
                    steer = 0.7
                    acceleration = 0.95
                else:
                    steer = 0
                    acceleration = 1

            #REGULAR PUCK CHASING
            if step > 43:
                    if home_x_location < puck_x_location:
                        offset = -0.4
                    else:
                        offset = 0.4

                    if stats.mode(detector_tracker[-2:])[0] == 1:
                        if north == 1:
                            print("Chasing the Puck!")
                            if puck_y_location > 0.1:
                                print("Next to the Puck!")
                                acceleration = 0.0
                                brake = True
                                if player_one_location[0] > 0:
                                    steer = 0.9
                                else:
                                    steer = -0.9
                            else:
                                if puck_x_location < 0:
                                    steer = -1
                                    acceleration = 0.7
                                    if puck_x_location < -0.4:
                                        drift = True
                                elif puck_x_location  > 0:
                                    steer = 1
                                    acceleration = 0.7
                                    if puck_x_location > 0.4:
                                        drift = True
                                else:
                                    steer = 0
                                    acceleration = 1

                        if north == 0:
                            print("Retreating to our end!")
                            if puck_x_location+offset< 0:
                                    steer = -1
                                    acceleration = 0.5
                                    if puck_x_location+offset < -0.4:
                                        drift = True
                            elif puck_x_location+offset > 0:
                                    steer = 1
                                    acceleration = 0.5
                                    if puck_x_location+offset > 0.4:
                                        drift = True
                            else:
                                    steer = 0
                                    acceleration = 1

                    elif stats.mode(detector_tracker[-2:])[0] == 0:
                        print("Engage Circle Mode!")
                        acceleration = 0.0
                        brake = True
                        if player_one_location[0] > 0:
                            steer = 0.8
                        else:
                            steer = -0.8

            print(stats.mode(detector_tracker[-3:])[0])
            action = {'acceleration': acceleration, 'brake': brake, 'drift': drift, 'nitro': nitro, 'rescue': rescue, 'steer': steer}

        if (self.player_id == 1):

            # IMPORTANT 3D LOCATIONS
            home_net_location = blue_net
            target_net_location = red_net
            player_two_location = player_info.kart.location

            # PLAYER TEMPORAL TRACKING
            step = len(player_two_location_x)
            player_two_location_x.append(player_two_location[0])
            player_two_location_y.append(player_two_location[2])
            current_vel = abs(max(player_info.kart.velocity))
            player_two_velocity.append(current_vel)

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
            detector = load_det_model()

            # PUCK
            puck_point_image = planner(TF.to_tensor(image)[None]).squeeze(0).cpu().detach().numpy()
            puck_visibility = detector(TF.to_tensor(image)[None]).squeeze(0).cpu().detach().numpy()

            puck_x_location = puck_point_image[0]
            puck_y_location = puck_point_image[1]
            puck_z_location = puck_visibility

            player_two_puck_x_average.append(puck_x_location)

            vol_check = []
            for i in player_two_puck_x_average[-10:]:
                vol_check.append(float(i))

            # CONTROLLER USING X-Y POINTS

            # SET ACTION DEFAULTS
            nitro = True
            rescue = False
            brake = False
            drift = False
            steer = 0
            buffer = 1
            acceleration = 0.2

            # DETERMINE NORTH SOUTH KART ROTATION
            if abs(convert_to_angles(player_info.kart.rotation)[0]) > 170:
                north = 0
            else:
                north = 1

            # DETERMINE EAST WEST KART ROATION (left is 90, right is -90)
            east_west = convert_to_angles(player_info.kart.rotation)[1]

            # BUILD OUT LOGIC FOR DECISIONS

            # LAUNCH CONTROL
            if step <= 41:
                if puck_x_location < 0:
                    steer = -0.6
                    acceleration = 1
                elif puck_x_location > 0:
                    steer = 0.6
                    acceleration = 1

            # REGULAR PUCK CHASING
            else:
                if puck_x_location < 0 and puck_x_location > -0.2:
                    steer = -1 * buffer
                    acceleration = 0.7
                elif puck_x_location > 0 and puck_x_location < 0.2:
                    steer = 1 * buffer
                    acceleration = 0.7

                # RESCUE MODE
                if abs(np.mean(vol_check[-5:])) > 0.2:
                    acceleration = 0.1
                    brake = False
                    steer = 1

            action = {'acceleration': acceleration, 'brake': brake, 'drift': drift, 'nitro': nitro, 'rescue': rescue, 'steer': steer}

        if (self.player_id == 2):

            #IMPORTANT 3D LOCATIONS
            home_net_location = red_net
            target_net_location = blue_net
            player_three_location = player_info.kart.location

            #PLAYER TEMPORAL TRACKING
            step = len(player_three_location_x)
            player_three_location_x.append(player_three_location[0])
            player_three_location_y.append(player_three_location[2])
            current_vel = abs(max(player_info.kart.velocity))
            player_three_velocity.append(current_vel)

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
            puck_point_image = planner(TF.to_tensor(image)[None]).squeeze(0).cpu().detach().numpy()
            puck_visibility = detector(TF.to_tensor(image)[None]).squeeze(0).cpu().detach().numpy()

            puck_x_location = puck_point_image[0]
            puck_y_location = puck_point_image[1]
            puck_z_location = puck_visibility

            player_three_puck_x_average.append(puck_x_location)

            vol_check = []
            for i in player_three_puck_x_average[-10:]:
                vol_check.append(float(i))

            #CONTROLLER USING X-Y POINTS
            #SET ACTION DEFAULTS
            nitro = True
            rescue = False
            brake = False
            drift = False
            steer = 0
            buffer = 1
            acceleration = 0.2

            #DETERMINE NORTH SOUTH KART ROTATION
            if abs(convert_to_angles(player_info.kart.rotation)[0]) > 170:
                north = 0
            else:
                north = 1

            #DETERMINE EAST WEST KART ROATION (left is 90, right is -90)
            east_west = convert_to_angles(player_info.kart.rotation)[1]

            #BUILD OUT LOGIC FOR DECISIONS

            #LAUNCH CONTROL
            if step <= 41:
                if puck_x_location < 0:
                    steer = -0.6
                    acceleration = 1
                elif puck_x_location > 0:
                    steer = 0.6
                    acceleration = 1

            #REGULAR PUCK CHASING
            else:
                if puck_x_location < 0 and puck_x_location > -0.2:
                    steer = -1*buffer
                    acceleration = 0.7
                elif puck_x_location > 0 and puck_x_location < 0.2:
                    steer = 1*buffer
                    acceleration = 0.7

            #RESCUE MODE
                if abs(np.mean(vol_check[-5:])) > 0.2:
                    acceleration = 0.1
                    brake = False
                    steer = 1

            action = {'acceleration': acceleration, 'brake': brake, 'drift': drift, 'nitro': nitro, 'rescue': rescue, 'steer': steer}

        if (self.player_id == 3):

            #IMPORTANT 3D LOCATIONS
            home_net_location = blue_net
            target_net_location = red_net
            player_four_location = player_info.kart.location

            #PLAYER TEMPORAL TRACKING
            step = len(player_three_location_x)
            player_four_location_x.append(player_four_location[0])
            player_three_location_y.append(player_four_location[2])
            current_vel = abs(max(player_info.kart.velocity))
            player_four_velocity.append(current_vel)

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
            puck_point_image = planner(TF.to_tensor(image)[None]).squeeze(0).cpu().detach().numpy()
            puck_visibility = detector(TF.to_tensor(image)[None]).squeeze(0).cpu().detach().numpy()

            puck_x_location = puck_point_image[0]
            puck_y_location = puck_point_image[1]
            puck_z_location = puck_visibility

            player_four_puck_x_average.append(puck_x_location)

            vol_check = []
            for i in player_four_puck_x_average[-10:]:
                vol_check.append(float(i))

            #CONTROLLER USING X-Y POINTS
            #SET ACTION DEFAULTS
            nitro = True
            rescue = False
            brake = False
            drift = False
            steer = 0
            buffer = 1
            acceleration = 0.2

            #DETERMINE NORTH SOUTH KART ROTATION
            if abs(convert_to_angles(player_info.kart.rotation)[0]) > 170:
                north = 0
            else:
                north = 1

            #DETERMINE EAST WEST KART ROATION (left is 90, right is -90)
            east_west = convert_to_angles(player_info.kart.rotation)[1]

            #BUILD OUT LOGIC FOR DECISIONS

            #LAUNCH CONTROL
            if step <= 41:
                if puck_x_location < 0:
                    steer = -0.6
                    acceleration = 1
                elif puck_x_location > 0:
                    steer = 0.6
                    acceleration = 1

            #REGULAR PUCK CHASING
            else:
                if puck_x_location < 0 and puck_x_location > -0.2:
                    steer = -1*buffer
                    acceleration = 0.7
                elif puck_x_location > 0 and puck_x_location < 0.2:
                    steer = 1*buffer
                    acceleration = 0.7

            #RESCUE MODE
                if abs(np.mean(vol_check[-5:])) > 0.2:
                    acceleration = 0.1
                    brake = False
                    steer = 1

            action = {'acceleration': acceleration, 'brake': brake, 'drift': drift, 'nitro': nitro, 'rescue': rescue, 'steer': steer}

        return action


