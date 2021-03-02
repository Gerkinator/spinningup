from spinup.algos.pytorch.ppo.ppo import ppo
import hydra


from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig



if __name__ == "__main__":
    config = DictConfig({})
    config['class'] = 'RoboticFetchPush.FetchPush'
    config['params'] = DictConfig({})
    config.params['reward_type'] = 'dense'
    env = hydra.utils.instantiate(config)

    logger_kwargs = {'output_dir': "./data",
                     'exp_name': "ppoTest"}

    ppo(env, logger_kwargs=logger_kwargs)