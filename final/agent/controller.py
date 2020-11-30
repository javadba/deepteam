import pystk


def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    current_y_location = 0
    current_x_location = 0

    aim_x_location = aim_point[0]
    aim_y_location = aim_point[1]

    action = pystk.Action()
    action.acceleration = 1 * (1-abs(aim_x_location))

    if aim_x_location < 0:
        action.steer = -1
    elif aim_x_location > 0:
        action.steer = 1
    else:
        action.steer = 0

    if aim_x_location < -0.4:
        action.drift=True
    elif aim_x_location > 0.4:
        action.drift=True
    else:
        action.drift=False

    if current_vel > 18 and abs(aim_x_location) > abs(0.5):
        action.acceleration=0
        action.brake=True

    if current_vel > 20:
        action.acceleration = 0

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """
    return action


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
