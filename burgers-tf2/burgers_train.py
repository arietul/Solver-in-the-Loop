# ----------------------------------------------------------------------------
#
# Phiflow Burgers equation solver framework
# Copyright 2020 Kiwon Um, Nils Thuerey
#
# This program is free software, distributed under the terms of the
# Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# Training (PyTorch version)
#
# ----------------------------------------------------------------------------

import os, sys, logging, argparse, pickle, glob, random, distutils.dir_util

log = logging.getLogger()
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

params = {}
parser = argparse.ArgumentParser(description='Parameter Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu',             default='0',               help='visible GPUs')
parser.add_argument('--cuda',            action='store_true',       help='enable CUDA for solver')
parser.add_argument('--train',           default=None,              help='training; will load data from this folder (set)')
parser.add_argument('--skip-ds',         action='store_true',       help='skip down-scaling; assume you have already saved')
parser.add_argument('--only-ds',         action='store_true',       help='exit after down-scaling and saving; use only for data pre-processing')
parser.add_argument('--log',             default=None,              help='path to a log file')
parser.add_argument('-s', '--scale',     default=4, type=int,       help='simulation scale for high-res')
parser.add_argument('-n', '--nsims',     default=10, type=int,      help='number of simulations')
parser.add_argument('-b', '--sbatch',    default=2, type=int,       help='size of a batch; when 10 simulations with the size of 5, 5 simulations are into two batches')
parser.add_argument('-t', '--simsteps',  default=200, type=int,     help='simulation steps; # of data samples (i.e. frames) per simulation')
parser.add_argument('-m', '--msteps',    default=2, type=int,       help='multi steps in training loss')
parser.add_argument('-e', '--epochs',    default=10, type=int,      help='training epochs')
parser.add_argument('--seed',            default=0, type=int,       help='seed for random number generator')
parser.add_argument('--noforce',         action='store_true',       help='no randomized external forces')
parser.add_argument('-l', '--len',       default=32, type=int,      help='length of the reference axis')  # FIXME: save and restore from the data
parser.add_argument('--dt',              default=1.0, type=float,   help='simulation time step size')
parser.add_argument('--model',           default='mars_moon',       help='(predefined) network model')
parser.add_argument('--lr',              default=1e-3, type=float,  help='start learning rate')
parser.add_argument('--adplr',           action='store_true',       help='turn on adaptive learning rate')
parser.add_argument('--resume',          default=-1, type=int,      help='resume training epochs')
parser.add_argument('--initpt',          default=None,              help='load initial model weights (warm start)')
parser.add_argument('--prept',           default=None,              help='load pre-trained weights (only for testing pre-trained supervised model; do not use for a warm start!)')
parser.add_argument('--pt',              default='/tmp/phiflow/pt', help='path to a pytorch output dir (model, logs, etc.)')
sys.argv += ['--' + p for p in params if isinstance(params[p], bool) and params[p]]
pargs = parser.parse_args()
params.update(vars(pargs))

os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']

import torch
import torch.nn as nn
import torch.nn.functional as F

from phi.flow import *

if params['log']:
    if params['resume']>0: params['log'] = os.path.splitext(params['log'])[0] + '_resume{:04d}'.format(params['resume']) + os.path.splitext(params['log'])[1]
    distutils.dir_util.mkpath(os.path.dirname(params['log']))
    log.addHandler(logging.FileHandler(params['log']))

if (params['nsims'] % params['sbatch']) != 0:
    params['nsims'] = (params['nsims']//params['sbatch'])*params['sbatch']
    log.info('Number of simulations is not divided by the batch size thus adjusted to {}'.format(params['nsims']))

log.info(params)
log.info('torch-{}'.format(torch.__version__))

random.seed(params['seed'])
np.random.seed(params['seed'])
torch.manual_seed(params['seed'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_feature(smokestates, forcestates):
    # input feature used for supervised version; drop the unused edges of the
    # staggered velocity grid making its dim same to the centered grid's
    return math.concat(
        [smokestates[j].velocity.staggered_tensor()[:, :-1:, :-1:, 0:2] for j in range(len(smokestates))] +
        [forcestates[j].velocity.staggered_tensor()[:, :-1:, :-1:, 0:2] for j in range(len(forcestates))],
        axis=-1
    )

def to_feature_noforce(smokestates):
    # input feature used for supervised version; drop the unused edges of the
    # staggered velocity grid making its dim same to the centered grid's
    return math.concat(
        [smokestates[j].velocity.staggered_tensor()[:, :-1:, :-1:, 0:2] for j in range(len(smokestates))],
        axis=-1
    )

def to_staggered(tensor_cen, box):
    return StaggeredGrid(math.pad(tensor_cen, ((0,0), (0,1), (0,1), (0,0))), box=box)


class ModelMercury(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 2, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)

class ModelMarsMoon(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.res_convs = nn.ModuleList()
        for _ in range(5):
            self.res_convs.append(nn.ModuleList([
                nn.Conv2d(32, 32, kernel_size=5, padding=2),
                nn.Conv2d(32, 32, kernel_size=5, padding=2),
            ]))
        self.output_conv = nn.Conv2d(32, 2, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.leaky_relu(self.initial_conv(x), negative_slope=0.3)
        for conv1, conv2 in self.res_convs:
            residual = x
            x = F.leaky_relu(conv1(x), negative_slope=0.3)
            x = conv2(x)
            x = F.leaky_relu(x + residual, negative_slope=0.3)
        return self.output_conv(x)

def downsample4xSMAC(tensor):
    return StaggeredGrid(tensor).downsample2x().downsample2x().staggered_tensor()

downVel = eval('downsample{}xSMAC'.format(params['scale']))

def lr_schedule(epoch, current_lr):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 10, 15, 20, 22 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = current_lr
    if   epoch == 23: lr *= 0.5
    elif epoch == 21: lr *= 1e-1
    elif epoch == 16: lr *= 1e-1
    elif epoch == 11: lr *= 1e-1
    return lr


@struct.definition()
class BurgersVelocitySMAC(BurgersVelocity):
    @struct.variable(dependencies=DomainState.domain, default=0)
    def velocity(self, velocity):
        return self.staggered_grid('velocity', velocity)

class BurgersTest(Burgers):
    def __init__(self, default_viscosity=0.1, viscosity=None, diffusion_substeps=1):
        Burgers.__init__(self, default_viscosity=default_viscosity, viscosity=viscosity, diffusion_substeps=diffusion_substeps)

    def step(self, v, dt=1.0, effects=()):
        return super().step(v=v, dt=dt, effects=effects)

    def step_with_f(self, v, f, dt=1.0):
        v_new = super().step(v=v, dt=dt)
        return v_new.copied_with(velocity=v_new.velocity + dt*f.velocity)

class PhifDataset():
    def __init__(self, dirpath, num_frames, num_sims=None, batch_size=1, print_fn=print, skip_preprocessing=False):
        self.dataSims      = sorted(glob.glob(dirpath + '/sim_0*'))[0:num_sims]
        self.pathsVel      = [ sorted(glob.glob(asim + '/velo_0*.npz')) for asim in self.dataSims ]
        self.pathsFrc      = [ sorted(glob.glob(asim + '/forc_0*.npz')) for asim in self.dataSims ]
        self.dataFrms      = [ np.arange(num_frames) for _ in self.dataSims ]  # NOTE: may contain different numbers of frames
        self.batchSize     = batch_size
        self.epoch         = None
        self.epochIdx      = 0
        self.batch         = None
        self.batchIdx      = 0
        self.step          = None
        self.stepIdx       = 0
        self.dataPreloaded = None
        self.printFn       = print_fn

        self.numOfSims    = num_sims
        self.numOfBatchs  = self.numOfSims//self.batchSize
        self.numOfFrames  = num_frames
        self.numOfSteps   = num_frames

        if not skip_preprocessing:
            self.printFn('Pre-processing: Loading data from {} = {} and save down-scaled data'.format(dirpath, self.dataSims))
            for j,asim in enumerate(self.dataSims):
                for i in range(num_frames):
                    v = downVel(read_zipped_array(self.pathsVel[j][i]))
                    f = downVel(read_zipped_array(self.pathsFrc[j][i]))
                    write_zipped_array(self.filenameToDownscaled(self.pathsVel[j][i]), v)
                    write_zipped_array(self.filenameToDownscaled(self.pathsFrc[j][i]), f)
                    self.printFn('Wrote {}'.format(self.filenameToDownscaled(self.pathsVel[j][i])))
                    self.printFn('Wrote {}'.format(self.filenameToDownscaled(self.pathsFrc[j][i])))

        self.printFn('Preload: Loading data from {} = {}'.format(dirpath, self.dataSims))
        self.dataPreloaded = {  # dataPreloaded['sim_key'][frame #][0=velocity, 1=force]
            asim: [
                (
                    read_zipped_array(self.filenameToDownscaled(self.pathsVel[j][i])),
                    read_zipped_array(self.filenameToDownscaled(self.pathsFrc[j][i])),
                ) for i in range(num_frames)
            ] for j,asim in enumerate(self.dataSims)
        }

        self.resolution = self.dataPreloaded[self.dataSims[0]][0][0].shape[1:3]  # [batch-size, y-size, x-size, dim]
        self.resolution = [v-1 for v in self.resolution]  # SMAC grid! calculate centered grid size
        # TODO: need a sanity check for resolution over all data

        self.dataStats = {
            'std': (
                # velocity
                (
                    np.std(np.concatenate([np.absolute(self.dataPreloaded[asim][i][0][...,0].reshape(-1)) for asim in self.dataSims for i in range(num_frames)])),  # vel[0]
                    np.std(np.concatenate([np.absolute(self.dataPreloaded[asim][i][0][...,1].reshape(-1)) for asim in self.dataSims for i in range(num_frames)])),  # vel[1]
                ),
                # force
                (
                    np.std(np.concatenate([np.absolute(self.dataPreloaded[asim][i][1][...,0].reshape(-1)) for asim in self.dataSims for i in range(num_frames)])),  # force[0]
                    np.std(np.concatenate([np.absolute(self.dataPreloaded[asim][i][1][...,1].reshape(-1)) for asim in self.dataSims for i in range(num_frames)])),  # force[1]
                ),
            )
        }
        self.printFn('Loaded {} samples'.format(self.numOfSims*self.numOfFrames))
        self.printFn(self.dataStats)

    def filenameToDownscaled(self, fname):
        return os.path.dirname(fname) + '/ds_' + os.path.basename(fname)

    def getInstance(self, sim_idx=0, frame=0):
        v0_hi = math.concat([self.dataPreloaded[self.dataSims[sim_idx+i]][frame][0] for i in range(self.batchSize)], axis=0)
        f0_hi = math.concat([self.dataPreloaded[self.dataSims[sim_idx+i]][frame][1] for i in range(self.batchSize)], axis=0)
        return [ v0_hi, f0_hi ]

    def newEpoch(self, exclude_tail=0, shuffle_data=True):
        self.numOfSteps = self.numOfFrames - exclude_tail
        simSteps = [ (asim, self.dataFrms[i][0:(len(self.dataFrms[i])-exclude_tail)]) for i,asim in enumerate(self.dataSims) ]
        sim_step_pair = []
        for i,_ in enumerate(simSteps):
            sim_step_pair += [ (i, astep) for astep in simSteps[i][1] ]  # (sim_idx, step) ...

        if shuffle_data: random.shuffle(sim_step_pair)
        self.epoch = [ list(sim_step_pair[i*self.numOfSteps:(i+1)*self.numOfSteps]) for i in range(self.batchSize*self.numOfBatchs) ]
        self.epochIdx += 1
        self.batchIdx = 0
        self.stepIdx = 0

    def nextBatch(self):  # batch size may be the number of simulations in a batch
        self.batchIdx += self.batchSize
        self.stepIdx = 0

    def nextStep(self):
        self.stepIdx += 1

    def getData(self, consecutive_frames, with_skip=1):
        v_hi = [
            math.concat([
                self.dataPreloaded[
                    self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]]  # sim_key
                ][
                    self.epoch[self.batchIdx+i][self.stepIdx][1]+j*with_skip  # steps
                ][0]            # velocity
                for i in range(self.batchSize)
            ], axis=0) for j in range(consecutive_frames+1)
        ]
        f_hi = [
            math.concat([
                self.dataPreloaded[
                    self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]]  # sim_key
                ][
                    self.epoch[self.batchIdx+i][self.stepIdx][1]+j*with_skip  # steps
                ][1]            # force
                for i in range(self.batchSize)
            ], axis=0) for j in range(consecutive_frames+1)
        ]
        return [ v_hi, f_hi ]

    def getPrevData(self, previous_frames, with_skip=1):  # NOTE: not in use; need to test
        v_hi = [
            math.concat([
                self.dataPreloaded[
                    self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]]
                ][
                    max([0, self.epoch[self.batchIdx+i][self.stepIdx][1]-j*with_skip])
                ][0]
                for i in range(self.batchSize)
            ], axis=0) for j in range(previous_frames)
        ]
        f_hi = [
            math.concat([
                self.dataPreloaded[
                    self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]]
                ][
                    max([0, self.epoch[self.batchIdx+i][self.stepIdx][1]-j*with_skip])
                ][1]
                for i in range(self.batchSize)
            ], axis=0) for j in range(previous_frames)
        ]
        return [ v_hi, f_hi ]


simulator_lo = BurgersTest()

dataset = PhifDataset(
    dirpath=params['train'],
    num_frames=params['simsteps'], num_sims=params['nsims'], batch_size=params['sbatch'],
    print_fn=log.info,
    skip_preprocessing=params['skip_ds']
)
if params['only_ds']: exit(0)

if params['prept']:
    with open(os.path.dirname(params['prept'])+'/stats.pickle', 'rb') as f: ld_stats = pickle.load(f)
    dataset.dataStats['in.std'] = ((ld_stats['in.std'][0], ld_stats['in.std'][1]),)
    dataset.dataStats['out.std'] = ld_stats['out.std']
    log.info(dataset.dataStats)

if params['resume']>0:
    with open(params['pt']+'/dataStats.pickle', 'rb') as f: dataset.dataStats = pickle.load(f)

dm_co = Domain(resolution=list(dataset.resolution), box=box([params['len']]*2), boundaries=PERIODIC)

st_co =   BurgersVelocitySMAC(dm_co, batch_size=params['sbatch'])
st_gt = [ BurgersVelocitySMAC(dm_co, batch_size=params['sbatch']) for _ in range(params['msteps']) ]  # ground truth velocities
st_fr = [ BurgersVelocitySMAC(dm_co, batch_size=params['sbatch']) for _ in range(params['msteps']) ]  # forces

if (params['train'] is None):
    log.info(params['train'])
    log.info('No pre-loadable training data path is given.')
    exit(0)

scene = Scene.create(params['train'], count=params['sbatch'], mkdir=False, copy_calling_script=False)

# Determine input channels based on noforce flag
if params['noforce']:
    in_channels = 2   # velocity only (u, v)
else:
    in_channels = 4   # velocity (u, v) + force (u, v)

# Create model
if params['model'] == 'mercury':
    model = ModelMercury(in_channels).to(device)
elif params['model'] == 'mars_moon':
    model = ModelMarsMoon(in_channels).to(device)
else:
    raise ValueError('Unknown model: {}'.format(params['model']))

log.info(model)

if params['prept']:
    log.info('load a pre-trained model: {}'.format(params['prept']))
    model.load_state_dict(torch.load(params['prept'], map_location=device))

if params['initpt']:
    log.info('load an initial model (warm start): {}'.format(params['initpt']))
    model.load_state_dict(torch.load(params['initpt'], map_location=device))

if params['resume']<1:
    [ params['pt'] and distutils.dir_util.mkpath(params['pt']) ]
    with open(params['pt']+'/dataStats.pickle', 'wb') as f: pickle.dump(dataset.dataStats, f)
else:
    loadpath = params['pt']+'/model_epoch{:04d}.pt'.format(params['resume'])
    log.info('load a resuming model (trained up to {} epoths): {}'.format(params['resume'], loadpath))
    model.load_state_dict(torch.load(loadpath, map_location=device))

# TensorBoard
from torch.utils.tensorboard import SummaryWriter
tb_writer = SummaryWriter(log_dir=params['pt']+'/summary/training')

current_lr = params['lr']
opt = torch.optim.Adam(model.parameters(), lr=current_lr)

i_st = 0
for j in range(params['epochs']):  # training
    dataset.newEpoch(exclude_tail=params['msteps'])
    if j<params['resume']:
        log.info('resume: skipping {} epoch'.format(j+1))
        i_st += dataset.numOfSteps*dataset.numOfBatchs
        continue

    current_lr = lr_schedule(j, current_lr) if params['adplr'] else params['lr']
    for param_group in opt.param_groups:
        param_group['lr'] = current_lr

    for ib in range(dataset.numOfBatchs):   # for each batch
        for i in range(dataset.numOfSteps):  # for each step
            adata = dataset.getData(consecutive_frames=params['msteps'], with_skip=1)
            st_co = st_co.copied_with(velocity=adata[0][0])
            if not params['noforce']: st_fr = [ st_fr[k].copied_with(velocity=adata[1][k  ]) for k in range(params['msteps']) ]
            st_gt = [ st_gt[k].copied_with(velocity=adata[0][k+1]) for k in range(params['msteps']) ]

            opt.zero_grad()

            # Multi-step prediction with correction
            tf_st_co_prd = []
            tf_cv_md = []
            for mi in range(params['msteps']):
                # Solver step
                if params['noforce']:
                    pred = simulator_lo.step(
                        v=st_co if mi==0 else tf_st_co_prd[-1],
                        dt=params['dt']
                    )
                else:
                    pred = simulator_lo.step_with_f(
                        v=st_co if mi==0 else tf_st_co_prd[-1],
                        f=st_fr[mi],
                        dt=params['dt']
                    )
                tf_st_co_prd.append(pred)

                # Build feature and run model
                if params['noforce']:
                    feature = to_feature_noforce(smokestates=[tf_st_co_prd[-1]])
                    in_std = [
                        *(dataset.dataStats['in.std' if 'in.std' in dataset.dataStats else 'std'][0]),  # velocity
                    ]
                else:
                    feature = to_feature(smokestates=[tf_st_co_prd[-1]], forcestates=[st_fr[mi]])
                    in_std = [
                        *(dataset.dataStats['in.std' if 'in.std' in dataset.dataStats else 'std'][0]),  # velocity
                        *(dataset.dataStats['in.std' if 'in.std' in dataset.dataStats else 'std'][1]),  # force
                    ]

                feature_normalized = feature / in_std
                out_std = dataset.dataStats['out.std' if 'out.std' in dataset.dataStats else 'std'][0]

                # NHWC -> NCHW for PyTorch conv
                model_input = torch.tensor(np.array(feature_normalized), dtype=torch.float32).permute(0, 3, 1, 2).to(device)
                model_out = model(model_input)
                # NCHW -> NHWC
                model_out_nhwc = model_out.permute(0, 2, 3, 1)
                model_out_denorm = model_out_nhwc * torch.tensor(out_std, dtype=torch.float32, device=device)

                correction = to_staggered(model_out_denorm.detach().cpu().numpy(), box=st_co.velocity.box)
                tf_cv_md.append(correction)

                tf_st_co_prd[-1] = tf_st_co_prd[-1].copied_with(velocity=tf_st_co_prd[-1].velocity + correction)

            # Compute loss
            loss_steps = []
            for mi in range(params['msteps']):
                gt_vel = torch.tensor(
                    np.array(st_gt[mi].velocity.staggered_tensor()),
                    dtype=torch.float32, device=device
                )
                pred_vel = torch.tensor(
                    np.array(tf_st_co_prd[mi].velocity.staggered_tensor()),
                    dtype=torch.float32, device=device
                )
                norm_std = torch.tensor(dataset.dataStats['std'][0], dtype=torch.float32, device=device)
                step_loss = torch.sum(((gt_vel - pred_vel) / norm_std)**2) / 2
                loss_steps.append(step_loss)

            total_loss = sum(loss_steps) / params['msteps']

            # NOTE: Because the simulation steps use PhiFlow (not differentiable through PyTorch),
            # we need to compute gradients only through the model output.
            # Re-run the model forward pass to get a differentiable loss.

            # Re-run forward passes with gradient tracking
            opt.zero_grad()
            diff_loss_total = torch.tensor(0.0, dtype=torch.float32, device=device)

            # We need to re-run the model forward for each step to get gradients
            st_co_curr = st_co
            for mi in range(params['msteps']):
                # Solver step (non-differentiable)
                if params['noforce']:
                    pred = simulator_lo.step(v=st_co_curr, dt=params['dt'])
                else:
                    pred = simulator_lo.step_with_f(v=st_co_curr, f=st_fr[mi], dt=params['dt'])

                # Build feature
                if params['noforce']:
                    feature = to_feature_noforce(smokestates=[pred])
                    in_std = [
                        *(dataset.dataStats['in.std' if 'in.std' in dataset.dataStats else 'std'][0]),
                    ]
                else:
                    feature = to_feature(smokestates=[pred], forcestates=[st_fr[mi]])
                    in_std = [
                        *(dataset.dataStats['in.std' if 'in.std' in dataset.dataStats else 'std'][0]),
                        *(dataset.dataStats['in.std' if 'in.std' in dataset.dataStats else 'std'][1]),
                    ]

                feature_normalized = feature / in_std
                out_std_vals = dataset.dataStats['out.std' if 'out.std' in dataset.dataStats else 'std'][0]

                # NHWC -> NCHW
                model_input = torch.tensor(np.array(feature_normalized), dtype=torch.float32).permute(0, 3, 1, 2).to(device)
                model_out = model(model_input)
                # NCHW -> NHWC
                model_out_nhwc = model_out.permute(0, 2, 3, 1)
                model_out_denorm = model_out_nhwc * torch.tensor(out_std_vals, dtype=torch.float32, device=device)

                # Pad to staggered grid shape
                model_out_padded = F.pad(model_out_denorm, (0, 0, 0, 1, 0, 1))  # pad spatial dims (H+1, W+1)

                # Get solver prediction and ground truth as tensors
                pred_vel_np = np.array(pred.velocity.staggered_tensor())
                pred_vel = torch.tensor(pred_vel_np, dtype=torch.float32, device=device)

                gt_vel_np = np.array(st_gt[mi].velocity.staggered_tensor())
                gt_vel = torch.tensor(gt_vel_np, dtype=torch.float32, device=device)

                norm_std = torch.tensor(dataset.dataStats['std'][0], dtype=torch.float32, device=device)

                corrected_vel = pred_vel + model_out_padded
                step_loss = torch.sum(((gt_vel - corrected_vel) / norm_std)**2) / 2
                diff_loss_total = diff_loss_total + step_loss

                # Update state for next multi-step
                correction_sg = to_staggered(model_out_denorm.detach().cpu().numpy(), box=st_co.velocity.box)
                st_co_curr = pred.copied_with(velocity=pred.velocity + correction_sg)

            diff_loss_total = diff_loss_total / params['msteps']
            diff_loss_total.backward()
            opt.step()

            l2 = diff_loss_total.item()

            # TensorBoard logging
            for mi, sl in enumerate(loss_steps):
                tb_writer.add_scalar('loss_step{:02d}'.format(mi), sl.item(), i_st)
            tb_writer.add_scalar('l2', l2, i_st)
            tb_writer.add_scalar('total_loss', l2, i_st)
            tb_writer.add_scalar('lr', current_lr, i_st)

            i_st += 1

            log.info('epoch {:03d}/{:03d}, batch {:03d}/{:03d}, step {:04d}/{:04d}: loss={}'.format(
                j+1, params['epochs'], ib+1, dataset.numOfBatchs, i+1, dataset.numOfSteps, l2
            ))
            dataset.nextStep()

        dataset.nextBatch()

    if j%10==9 or j==0: torch.save(model.state_dict(), params['pt']+'/model_epoch{:04d}.pt'.format(j+1))

tb_writer.close()
torch.save(model.state_dict(), params['pt']+'/model.pt')
