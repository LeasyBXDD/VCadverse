from util.config import Config
from util.mytorch import same_seeds
from agent.inferencer import Inferencer

# load config file
config_path = 'config/train_again-c4s.yaml'
config = Config(config_path)

# hand-crafted argparser
dsp_config_path = 'config/preprocess.yaml'
dsp_config = Config(dsp_config_path)
load = 'checkpoints/again/c4s/steps_100000.pth'
args = {
    'dsp_config': dsp_config,
    'load': load
}
args = Config(args)

# build inferencer
inferencer = Inferencer(config=config, args=args, trust_repo=True)

# set paths and parameters
source_path = 'data/wav48/p225/p225_001.wav'
target_path = 'data/wav48/p226/p226_001.wav'
out_path = 'data/generated/'
seglen = None

# run inference
inferencer.inference(source_path=source_path, target_path=target_path, out_path=out_path, seglen=seglen)