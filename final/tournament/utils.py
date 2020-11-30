import pystk
import numpy as np
import torchvision.transforms.functional as TF
import statistics

TRACK_OFFSET = 15

class Player:
    def __init__(self, player, team=0):
        self.player = player
        self.team = team

    @property
    def config(self):
        return pystk.PlayerConfig(controller=pystk.PlayerConfig.Controller.PLAYER_CONTROL, kart=self.player.kart, team=self.team)
    
    def __call__(self, image, player_info):
        return self.player.act(image, player_info)


class Tournament:
    _singleton = None

    def __init__(self, players, screen_width=400, screen_height=300, track='icy_soccer_field'):
        assert Tournament._singleton is None, "Cannot create more than one Tournament object"
        Tournament._singleton = self

        self.graphics_config = pystk.GraphicsConfig.hd()
        self.graphics_config.screen_width = screen_width
        self.graphics_config.screen_height = screen_height
        pystk.init(self.graphics_config)

        self.race_config = pystk.RaceConfig(num_kart=len(players), track=track, mode=pystk.RaceConfig.RaceMode.SOCCER)
        self.race_config.players.pop()
        
        self.active_players = []
        for p in players:
            if p is not None:
                self.race_config.players.append(p.config)
                self.active_players.append(p)
        
        self.k = pystk.Race(self.race_config)

        self.k.start()
        self.k.step()


    @staticmethod
    def _to_image(x, proj, view):
        p = proj @ view @ np.array(list(x) + [1])
        return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

    def play(self, save=None, max_frames=50):
        state = pystk.WorldState()
        track = pystk.Track()
        if save is not None:
            import PIL.Image
            import os
            if not os.path.exists(save):
                os.makedirs(save)

        for t in range(max_frames):
            print('\rframe %d' % t, end='\r')

            state.update()
            #print(state.soccer.ball.location)
            #print(state.soccer.goal_line)

            #goal_line = False
            #opposing_goal = state.soccer.goal_line[:-1]
            #goal_target = []
            #if goal_line == True:
             #   goal_target = [statistics.mean([opposing_goal[0][0][0], opposing_goal[0][1][0]]), statistics.mean([opposing_goal[0][0][1], opposing_goal[0][1][1]]),statistics.mean([opposing_goal[0][0][2], opposing_goal[0][1][2]]), ]

            list_actions = []
            for i, p in enumerate(self.active_players):
                kart = state.players[i].kart
                aim_point_world = state.soccer.ball.location
                proj = np.array(state.players[i].camera.projection).T
                view = np.array(state.players[i].camera.view).T
                aim_point_image = self._to_image(aim_point_world, proj, view)
                player = state.players[i]
                image = np.array(self.k.render_data[i].image)
                
                action = pystk.Action()
                player_action = p(image, player)
                for a in player_action:
                    setattr(action, a, player_action[a])
                
                list_actions.append(action)

                if save is not None:
                    PIL.Image.fromarray(image).save(os.path.join(save, 'player%02d_%05d.png' % (i, t)))
                    dest = os.path.join(save, 'player%02d_%05d' % (i, t))
                    with open(dest + '.csv', 'w') as f:
                        f.write('%0.1f,%-0.1f' % tuple(aim_point_image))

            s = self.k.step(list_actions)
            if not s:  # Game over
                break

        if save is not None:
            import subprocess
            for i, p in enumerate(self.active_players):
                dest = os.path.join(save, 'player%02d' % i)
                output = save + '_player%02d.mp4' % i
                subprocess.call(['ffmpeg', '-y', '-framerate', '10', '-i', dest + '_%05d.png', output])


        if hasattr(state, 'soccer'):
            return state.soccer.score
        return state.soccer_score

    def close(self):
        self.k.stop()
        del self.k
