# ----------------------------------------------------------------------------
#
# Phiflow Karman vortex solver framework
# Copyright 2020-2021 Kiwon Um, Nils Thuerey
#
# This program is free software, distributed under the terms of the
# Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# Training
#
# ----------------------------------------------------------------------------

import os, sys, logging, argparse, pickle, glob, random, distutils.dir_util

log = logging.getLogger()
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

params = {}
parser = argparse.ArgumentParser(description='Parameter Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu',            default='0',               help='visible GPUs')
parser.add_argument('--train',          default=None,              help='training; will load data from this simulation folder (set) and save down-sampled files')
parser.add_argument('--skip-ds',        action='store_true',       help='skip down-scaling; assume you have already saved')
parser.add_argument('--only-ds',        action='store_true',       help='exit after down-scaling and saving; use only for data pre-processing')
parser.add_argument('--log',            default=None,              help='path to a log file')
parser.add_argument('-s', '--scale',    default=4, type=int,       help='simulation scale for high-res')
parser.add_argument('-n', '--nsims',    default=1, type=int,       help='number of simulations')
parser.add_argument('-b', '--sbatch',   default=1, type=int,       help='size of a batch; when 10 simulations with the size of 5, 5 simulations are into two batches')
parser.add_argument('-t', '--simsteps', default=1500, type=int,    help='simulation steps; # of data samples (i.e. frames) per simulation')
parser.add_argument('-m', '--msteps',   default=2, type=int,       help='multi steps in training loss')
parser.add_argument('-e', '--epochs',   default=10, type=int,      help='training epochs')
parser.add_argument('--seed',           default=None, type=int,    help='seed for random number generator')
parser.add_argument('-r', '--res',      default=32, type=int,      help='target (i.e., low-res) resolution') # FIXME: save and restore from the data
parser.add_argument('-l', '--len',      default=100, type=int,     help='length of the reference axis')      # FIXME: save and restore from the data
parser.add_argument('--model',          default='mars_moon',       help='(predefined) network model')
parser.add_argument('--reg-loss',       action='store_true',       help='turn on regularization loss')
parser.add_argument('--lr',             default=1e-3, type=float,  help='start learning rate')
parser.add_argument('--adplr',          action='store_true',       help='turn on adaptive learning rate')
parser.add_argument('--clip-grad',      action='store_true',       help='turn on clip gradients')
parser.add_argument('--resume',         default=-1, type=int,      help='resume training epochs')
parser.add_argument('--initpt',         default=None,              help='load initial model weights (warm start)')
parser.add_argument('--prept',          default=None,              help='load pre-trained weights (only for testing pre-trained supervised model; do not use for a warm start!)')
parser.add_argument('--pt',             default='/tmp/phiflow/pt', help='path to a pytorch output dir (model, logs, etc.)')
sys.argv += ['--' + p for p in params if isinstance(params[p], bool) and params[p]]
pargs = parser.parse_args()
params.update(vars(pargs))

os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']

from phi.physics._boundaries import Domain, OPEN, STICKY as CLOSED
from phi.torch.flow import *

import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    log.info('{} GPUs available'.format(gpu_count))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(params['seed'])
np.random.seed(params['seed'])
torch.manual_seed(params['seed'])
if torch.cuda.is_available(): torch.cuda.manual_seed_all(params['seed'])

if params['resume']>0 and params['log']:
    params['log'] = os.path.splitext(params['log'])[0] + '_resume{:04d}'.format(params['resume']) + os.path.splitext(params['log'])[1]

if params['log']:
    distutils.dir_util.mkpath(os.path.dirname(params['log']))
    log.addHandler(logging.FileHandler(params['log']))

if (params['nsims'] % params['sbatch']) != 0:
    params['nsims'] = (params['nsims']//params['sbatch'])*params['sbatch']
    log.info('Number of simulations is not divided by the batch size thus adjusted to {}'.format(params['nsims']))

log.info(params)
log.info('torch-{}'.format(torch.__version__))

class ModelMercury(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 2, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class ModelMarsMoon(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        # 5 residual blocks, each with 2 conv layers
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
        x = self.output_conv(x)
        return x

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


class KarmanFlow():
    def __init__(self, domain):
        self.domain = domain

        shape_v = self.domain.staggered_grid(0).vector['y'].shape
        vel_yBc = np.zeros(shape_v.sizes)
        vel_yBc[0:2, 0:vel_yBc.shape[1]-1] = 1.0
        vel_yBc[0:vel_yBc.shape[0], 0:1] = 1.0
        vel_yBc[0:vel_yBc.shape[0], -1:] = 1.0
        self.vel_yBc = math.tensor(vel_yBc, shape_v)
        self.vel_yBcMask = math.tensor(np.copy(vel_yBc), shape_v) # warning, only works for 1s, otherwise setup/scale

        self.inflow = self.domain.scalar_grid(Box[5:10, 25:75])         # TODO: scale with domain if necessary!
        self.obstacles = [Obstacle(Sphere(center=[50, 50], radius=10))] # TODO: scale with domain if necessary!

    def step(self, density_in, velocity_in, re, res, buoyancy_factor=0, dt=1.0, make_input_divfree=False, make_output_divfree=True): #, conserve_density=True):
        velocity = velocity_in
        density = density_in

        # apply viscosity
        velocity = phi.flow.diffuse.explicit(field=velocity, diffusivity=1.0/re*dt*res*res, dt=dt)
        vel_x = velocity.vector['x']
        vel_y = velocity.vector['y']

        # apply velocity BCs, only y for now; velBCy should be pre-multiplied
        vel_y = vel_y*(1.0 - self.vel_yBcMask) + self.vel_yBc
        velocity = self.domain.staggered_grid(phi.math.stack([vel_y.data, vel_x.data], channel('vector')))

        pressure = None
        if make_input_divfree:
            velocity, pressure = fluid.make_incompressible(velocity, self.obstacles)

        # --- Advection ---
        density = advect.semi_lagrangian(density+self.inflow, velocity, dt=dt)
        velocity = advected_velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)
        # if conserve_density and self.domain.boundaries['accessible_extrapolation'] == math.extrapolation.ZERO:  # solid boundary
        #     density = field.normalize(density, self.density)

        # --- Pressure solve ---
        if make_output_divfree:
            velocity, pressure = fluid.make_incompressible(velocity, self.obstacles)

        self.solve_info = {
            'pressure': pressure,
            'advected_velocity': advected_velocity,
        }

        return [density, velocity]

class PhifDataset():
    def __init__(self, domain, dirpath, num_frames, num_sims=None, batch_size=1, print_fn=print, skip_preprocessing=False):
        self.dataSims      = sorted(glob.glob(dirpath + '/sim_0*'))[0:num_sims]
        self.pathsDen      = [ sorted(glob.glob(asim + '/dens_0*.npz')) for asim in self.dataSims ]
        self.pathsVel      = [ sorted(glob.glob(asim + '/velo_0*.npz')) for asim in self.dataSims ]
        self.dataFrms      = [ np.arange(num_frames) for _ in self.dataSims ] # NOTE: may contain different numbers of frames
        self.batchSize     = batch_size
        self.epoch         = None
        self.epochIdx      = 0
        self.batch         = None
        self.batchIdx      = 0
        self.step          = None
        self.stepIdx       = 0
        self.dataPreloaded = None
        self.printFn       = print_fn
        self.domain        = domain # phiflow: target domain (i.e., low-res.)

        self.numOfSims    = num_sims
        self.numOfBatchs  = self.numOfSims//self.batchSize
        self.numOfFrames  = num_frames
        self.numOfSteps   = num_frames

        if not skip_preprocessing:
            self.printFn('Pre-processing: Loading data from {} = {} and save down-scaled data'.format(dirpath, self.dataSims))
            for j,asim in enumerate(self.dataSims):
                for i in range(num_frames):
                    if not os.path.isfile(self.filenameToDownscaled(self.pathsDen[j][i])):
                        d = phi.field.read(file=self.pathsDen[j][i]).at(self.domain.scalar_grid())
                        phi.field.write(field=d, file=self.filenameToDownscaled(self.pathsDen[j][i]))
                        self.printFn('Wrote {}'.format(self.filenameToDownscaled(self.pathsDen[j][i])))
                    if not os.path.isfile(self.filenameToDownscaled(self.pathsVel[j][i])):
                        v = phi.field.read(file=self.pathsVel[j][i]).at(self.domain.staggered_grid())
                        phi.field.write(field=v, file=self.filenameToDownscaled(self.pathsVel[j][i]))
                        self.printFn('Wrote {}'.format(self.filenameToDownscaled(self.pathsVel[j][i])))

        self.printFn('Preload: Loading data from {} = {}'.format(dirpath, self.dataSims))
        self.dataPreloaded = {  # dataPreloaded['sim_key'][frame number][0=density, 1=x-velocity, 2=y-velocity]
            asim: [
                (
                    np.expand_dims(phi.field.read(file=self.filenameToDownscaled(self.pathsDen[j][i])).values.numpy(('y', 'x')),             axis=0), # density
                    np.expand_dims(phi.field.read(file=self.filenameToDownscaled(self.pathsVel[j][i])).vector['x'].values.numpy(('y', 'x')), axis=0), # x-velocity
                    np.expand_dims(phi.field.read(file=self.filenameToDownscaled(self.pathsVel[j][i])).vector['y'].values.numpy(('y', 'x')), axis=0), # y-velocity
                ) for i in range(num_frames)
            ] for j,asim in enumerate(self.dataSims)
        }                       # for each, keep shape=[batch-size, res-y, res-x]
        assert len(self.dataPreloaded[self.dataSims[0]][0][0].shape)==3, 'Data shape is wrong.'
        assert len(self.dataPreloaded[self.dataSims[0]][0][1].shape)==3, 'Data shape is wrong.'
        assert len(self.dataPreloaded[self.dataSims[0]][0][2].shape)==3, 'Data shape is wrong.'

        self.dataStats = {
            'std': (
                np.std(np.concatenate([np.absolute(self.dataPreloaded[asim][i][0].reshape(-1)) for asim in self.dataSims for i in range(num_frames)], axis=-1)), # density
                np.std(np.concatenate([np.absolute(self.dataPreloaded[asim][i][1].reshape(-1)) for asim in self.dataSims for i in range(num_frames)], axis=-1)), # x-velocity
                np.std(np.concatenate([np.absolute(self.dataPreloaded[asim][i][2].reshape(-1)) for asim in self.dataSims for i in range(num_frames)], axis=-1)), # y-velocity
            )
        }

        self.extConstChannelPerSim = {} # extConstChannelPerSim['sim_key'][0=first channel, ...]; for now, only Reynolds Nr.
        num_of_ext_channel = 1
        for asim in self.dataSims:
            with open(asim+'/params.pickle', 'rb') as f:
                sim_params = pickle.load(f)
                self.extConstChannelPerSim[asim] = [ sim_params['re'] ] # Reynolds Nr.

        self.dataStats.update({
            'ext.std': [
                np.std([np.absolute(self.extConstChannelPerSim[asim][i]) for asim in self.dataSims]) for i in range(num_of_ext_channel) # Reynolds Nr
            ]
        })
        self.printFn(self.dataStats)

    def filenameToDownscaled(self, fname):
        return os.path.dirname(fname) + '/ds_' + os.path.basename(fname)

    def getInstance(self, sim_idx=0, frame=0):
        d0_hi = math.concat([self.dataPreloaded[self.dataSims[sim_idx+i]][frame][0] for i in range(self.batchSize)], axis=0)
        u0_hi = math.concat([self.dataPreloaded[self.dataSims[sim_idx+i]][frame][1] for i in range(self.batchSize)], axis=0)
        v0_hi = math.concat([self.dataPreloaded[self.dataSims[sim_idx+i]][frame][2] for i in range(self.batchSize)], axis=0)
        return [d0_hi, u0_hi, v0_hi] # TODO: additional channels

    def newEpoch(self, exclude_tail=0, shuffle_data=True):
        self.numOfSteps = self.numOfFrames - exclude_tail
        sim_frames = [ (asim, self.dataFrms[i][0:(len(self.dataFrms[i])-exclude_tail)]) for i,asim in enumerate(self.dataSims) ]
        sim_frame_pairs = []
        for i,_ in enumerate(sim_frames):
            sim_frame_pairs += [ (i, aframe) for aframe in sim_frames[i][1] ] # [(sim_idx, frame_number), ...]

        if shuffle_data: random.shuffle(sim_frame_pairs)
        self.epoch = [ list(sim_frame_pairs[i*self.numOfSteps:(i+1)*self.numOfSteps]) for i in range(self.batchSize*self.numOfBatchs) ]
        self.epochIdx += 1
        self.batchIdx = 0
        self.stepIdx = 0

    def nextBatch(self):        # batch size may be the number of simulations in a batch
        self.batchIdx += self.batchSize
        self.stepIdx = 0

    def nextStep(self):
        self.stepIdx += 1

    def getData(self, consecutive_frames, with_skip=1):
        d_hi = [
            np.concatenate([
                self.dataPreloaded[
                    self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]] # sim_key
                ][
                    self.epoch[self.batchIdx+i][self.stepIdx][1]+j*with_skip # frames
                ][0]
                for i in range(self.batchSize)
            ], axis=0) for j in range(consecutive_frames+1)
        ]
        u_hi = [
            np.concatenate([
                self.dataPreloaded[
                    self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]] # sim_key
                ][
                    self.epoch[self.batchIdx+i][self.stepIdx][1]+j*with_skip # frames
                ][1]
                for i in range(self.batchSize)
            ], axis=0) for j in range(consecutive_frames+1)
        ]
        v_hi = [
            np.concatenate([
                self.dataPreloaded[
                    self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]] # sim_key
                ][
                    self.epoch[self.batchIdx+i][self.stepIdx][1]+j*with_skip # frames
                ][2]
                for i in range(self.batchSize)
            ], axis=0) for j in range(consecutive_frames+1)
        ]
        ext = [
            self.extConstChannelPerSim[
                self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]]
            ][0] for i in range(self.batchSize)
        ]
        return [d_hi, u_hi, v_hi, ext]

    def getPrevData(self, previous_frames, with_skip=1): # NOTE: not in use; need to test
        d_hi = [
            math.concat([
                self.dataPreloaded[
                    self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]]
                ][
                    max([0, self.epoch[self.batchIdx+i][self.stepIdx][1]-j*with_skip])
                ][0]
                for i in range(self.batchSize)
            ], axis=0) for j in range(previous_frames)
        ]
        u_hi = [
            math.concat([
                self.dataPreloaded[
                    self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]]
                ][
                    max([0, self.epoch[self.batchIdx+i][self.stepIdx][1]-j*with_skip])
                ][1]
                for i in range(self.batchSize)
            ], axis=0) for j in range(previous_frames)
        ]
        v_hi = [
            math.concat([
                self.dataPreloaded[
                    self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]]
                ][
                    max([0, self.epoch[self.batchIdx+i][self.stepIdx][1]-j*with_skip])
                ][2]
                for i in range(self.batchSize)
            ], axis=0) for j in range(previous_frames)
        ]
        # TODO: additional channels
        return [d_hi, v_hi]


domain  = Domain(y=params['res']*2, x=params['res'], bounds=Box[0:params['len']*2, 0:params['len']], boundaries=OPEN)
simulator_lo = KarmanFlow(domain=domain)

dataset = PhifDataset(
    domain=domain,
    dirpath=params['train'],
    num_frames=params['simsteps'], num_sims=params['nsims'], batch_size=params['sbatch'],
    print_fn=log.info,
    skip_preprocessing=params['skip_ds']
)
if params['only_ds']: exit(0)

if params['prept']:
    with open(os.path.dirname(params['prept'])+'/stats.pickle', 'rb') as f: ld_stats = pickle.load(f)
    dataset.dataStats['in.std'] = (ld_stats['in.std'][0], (ld_stats['in.std'][1], ld_stats['in.std'][2]))
    dataset.dataStats['out.std'] = ld_stats['out.std']
    log.info(dataset.dataStats)

if params['resume']>0:
    with open(params['pt']+'/dataStats.pickle', 'rb') as f: dataset.dataStats = pickle.load(f)

if (params['train'] is None):
    log.info(params['train'])
    log.info('No pre-loadable training data path is given.')
    exit(0)

from torch.utils.tensorboard import SummaryWriter
tb_writer = SummaryWriter(log_dir=params['pt']+'/summary/training')

# model
model_classes = {
    'mercury': ModelMercury,
    'mars_moon': ModelMarsMoon,
}
model = model_classes[params['model']](in_channels=3).to(device)
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
    model.load_state_dict(torch.load(params['pt']+'/model_epoch{:04d}.pt'.format(params['resume']), map_location=device))

opt = torch.optim.Adam(model.parameters(), lr=params['lr'])

def to_feature(dens_vel_grid_array, ext_const_channel):
    # drop the unused edges of the staggered velocity grid making its dim same to the centered grid's
    return math.stack(
        [
            dens_vel_grid_array[1].vector['x'].x[:-1].values,         # u
            dens_vel_grid_array[1].vector['y'].y[:-1].values,         # v
            math.ones(dens_vel_grid_array[0].shape)*ext_const_channel # Re
        ],
        math.channel('channels')
    )

def to_staggered(torch_tensor, domain):
    # torch_tensor is NCHW: [batch, 2, y, x]; convert back to NHWC: [batch, y, x, 2]
    torch_tensor_nhwc = torch_tensor.permute(0, 2, 3, 1)
    return domain.staggered_grid(
        math.stack(
            [
                math.tensor(F.pad(torch_tensor_nhwc[..., 1], (0, 0, 0, 1)), math.batch('batch'), math.spatial('y, x')), # v
                math.tensor(F.pad(torch_tensor_nhwc[..., 0], (0, 1, 0, 0)), math.batch('batch'), math.spatial('y, x')), # u
            ], math.channel('vector')
        )
    )

def train_step(pf_in_dens_gt, pf_in_velo_gt, pf_in_Re, i_step):
    opt.zero_grad()

    pf_co_prd, pf_cv_md = [], [] # predicted states with correction, inferred velocity corrections
    for i in range(params['msteps']):
        # solver step
        pf_co_prd += [
            simulator_lo.step(
                density_in=pf_in_dens_gt[0] if i==0 else pf_co_prd[-1][0],
                velocity_in=pf_in_velo_gt[0] if i==0 else pf_co_prd[-1][1],
                re=pf_in_Re,
                res=params['res'],
            )
        ]       # pf_co_prd: [[density1, velocity1], [density2, velocity2], ...]

        # prediction (correction)
        model_input = to_feature(pf_co_prd[-1], pf_in_Re)
        model_input /= math.tensor([dataset.dataStats['std'][1], dataset.dataStats['std'][2], dataset.dataStats['ext.std'][0]], channel('channels')) # [u, v, Re]
        model_input_native = model_input.native(['batch', 'y', 'x', 'channels']) # NHWC
        model_input_nchw = model_input_native.permute(0, 3, 1, 2)               # NCHW
        model_out_nchw = model(model_input_nchw)                                 # NCHW
        model_out_nhwc = model_out_nchw.permute(0, 2, 3, 1)                      # NHWC
        model_out_nhwc = model_out_nhwc * torch.tensor([dataset.dataStats['std'][1], dataset.dataStats['std'][2]], device=device) # [u, v]
        pf_cv_md += [ to_staggered(model_out_nhwc.permute(0, 3, 1, 2), domain) ] # pf_cv_md: [velocity_correction1, velocity_correction2, ...]

        pf_co_prd[-1][1] = pf_co_prd[-1][1] + pf_cv_md[-1]

    # loss computation
    loss_steps_x = [
        torch.sum(
            (
                pf_in_velo_gt[i+1].vector['x'].values.native(('batch', 'y', 'x'))
                - pf_co_prd[i][1].vector['x'].values.native(('batch', 'y', 'x'))
            )**2
        ) / 2 / dataset.dataStats['std'][1]**2
        for i in range(params['msteps'])
    ]
    loss_steps_x_sum = torch.sum(torch.stack(loss_steps_x))

    loss_steps_y = [
        torch.sum(
            (
                pf_in_velo_gt[i+1].vector['y'].values.native(('batch', 'y', 'x'))
                - pf_co_prd[i][1].vector['y'].values.native(('batch', 'y', 'x'))
            )**2
        ) / 2 / dataset.dataStats['std'][2]**2
        for i in range(params['msteps'])
    ]
    loss_steps_y_sum = torch.sum(torch.stack(loss_steps_y))

    loss = (loss_steps_x_sum + loss_steps_y_sum)/params['msteps']

    i_step_val = int(math.to_int64(i_step).native())
    for i,a_step_loss in enumerate(loss_steps_x): tb_writer.add_scalar('loss_each_step_vel_x{:02d}'.format(i+1), a_step_loss.item(), i_step_val)
    for i,a_step_loss in enumerate(loss_steps_y): tb_writer.add_scalar('loss_each_step_vel_y{:02d}'.format(i+1), a_step_loss.item(), i_step_val)
    tb_writer.add_scalar('sum_steps_loss', loss.item(), i_step_val)

    total_loss = loss
    if params['reg_loss']:
        reg_loss = sum(torch.sum(p**2) / 2 for p in model.parameters())
        total_loss = total_loss + reg_loss
        tb_writer.add_scalar('loss_regularization', reg_loss.item(), i_step_val)

    tb_writer.add_scalar('loss', total_loss.item(), i_step_val)

    total_loss.backward()
    opt.step()

    return math.tensor(total_loss.item())

i_st = 0
for j in range(params['epochs']): # training
    dataset.newEpoch(exclude_tail=params['msteps'])
    if j<params['resume']:
        log.info('resume: skipping {} epoch'.format(j+1))
        i_st += dataset.numOfSteps*dataset.numOfBatchs
        continue

    for ib in range(dataset.numOfBatchs):   # for each batch
        for i in range(dataset.numOfSteps): # for each step
            # adata: [[dens0, dens1, ...], [x-velo0, x-velo1, ...], [y-velo0, y-velo1, ...], [ReynoldsNr(s)]]
            adata = dataset.getData(consecutive_frames=params['msteps'], with_skip=1)
            dens_gt = [         # [density0:CenteredGrid, density1, ...]
                domain.scalar_grid(
                    math.tensor(adata[0][k], math.batch('batch'), math.spatial('y, x'))
                ) for k in range(params['msteps']+1)
            ]
            velo_gt = [         # [velocity0:StaggeredGrid, velocity1, ...]
                domain.staggered_grid(
                    math.stack(
                        [
                            math.tensor(adata[2][k], math.batch('batch'), math.spatial('y, x')),
                            math.tensor(adata[1][k], math.batch('batch'), math.spatial('y, x')),
                        ], math.channel('vector')
                    )
                ) for k in range(params['msteps']+1)
            ]
            re_nr = math.tensor(adata[3], math.batch('batch'))

            l2 = train_step(dens_gt, velo_gt, re_nr, math.tensor(i_st))

            i_st += 1

            log.info('epoch {:03d}/{:03d}, batch {:03d}/{:03d}, step {:04d}/{:04d}: loss={}'.format(
                j+1, params['epochs'], ib+1, dataset.numOfBatchs, i+1, dataset.numOfSteps, l2
            ))
            dataset.nextStep()

        dataset.nextBatch()

    if j%10==9: torch.save(model.state_dict(), params['pt']+'/model_epoch{:04d}.pt'.format(j+1))

tb_writer.close()
torch.save(model.state_dict(), params['pt']+'/model.pt')
