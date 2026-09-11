encode_vp.py
100%
import os
import sys
import yaml
import subprocess
import argparse
import logging
import shutil
import time
import copy
import itertools
import torch as th
import torch.nn as nn
from torch import Tensor
from typing import Any, Dict, List, Literal, Optional, OrderedDict, Tuple, Union
from dataclasses import dataclass, field, fields
from PIL import Image
from glob import glob
from torchvision.transforms.functional import to_tensor
from utils.misc import get_best_device, ARMINT, FIXED_POINT_FRACTIONAL_MULT, TrainingExitCode, POSSIBLE_DEVICE, DescriptorOverfitter, DescriptorNN
from utils.helpers import pad_image
from utils.timer import Timer
from models.prefitter import PrefitterParameter, Prefitter
from models.overfitter import OverfitterParameter, OverFitter
from encoding_management.presets import AVAILABLE_PRESETS, Preset, TrainerPhase
from bitstream.encode import fnlic_encode


os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


@dataclass
class EncoderManager():
    preset_name: str
    start_lr: float = 1e-2
    n_loops: int = 1
    n_itr: int = int(1e5)
    
    preset: Preset = field(init=False)
    idx_best_loop: int = field(default=0, init=False)
    best_loss: float = field(default=1e6, init=False)
    loop_counter: int = field(default=0, init=False)
    loop_start_time: float = field(default=0., init=False)
    iterations_counter: int = field(default=0, init=False)
    total_training_time_sec: float = field(default=0.0, init=False)
    phase_idx: int = field(default=0, init=False)
    warm_up_done: bool = field(default=False, init=False)

    def __post_init__(self):
        assert self.preset_name in AVAILABLE_PRESETS, f'Preset {self.preset_name} not found.'
        self.preset = AVAILABLE_PRESETS.get(self.preset_name)(start_lr= self.start_lr, n_itr_per_phase=self.n_itr)
        flag_quantize_model = False
        for training_phase in self.preset.all_phases:
            if training_phase.quantize_model:
                flag_quantize_model = True
        assert flag_quantize_model, 'Preset must include quantization phase.'

    def record_beaten(self, candidate_loss: float) -> bool:
        return candidate_loss < self.best_loss

    def set_best_loss(self, new_best_loss: float):
        self.best_loss = new_best_loss
        self.idx_best_loop = self.loop_counter
    
    def pretty_string(self) -> str:
        s = 'EncoderManager value:\n--------------------------\n'
        for k in fields(self):
            if k.name == 'preset': continue
            s += f'{k.name:<25}: {str(getattr(self, k.name)):<80}\n'
        return s + '\n'

@dataclass(kw_only=True)
class LossFunctionOutput():
    loss: Optional[float] = None
    rate_nn_bpd: Optional[float] = None
    rate_latent_bpd: Optional[float] = None
    rate_img_bpd: Optional[float] = None

@dataclass
class EncoderOutput():
    latent_bpd: Tensor
    img_bpd: Tensor
    additional_data: Dict[str, Any] = field(default_factory = lambda: {})

@dataclass
class EncoderLogs(LossFunctionOutput):
    loss_function_output: LossFunctionOutput
    encoder_output: EncoderOutput
    original_frame: Tensor
    detailed_rate_nn: DescriptorOverfitter
    quantization_param_nn: DescriptorOverfitter
    encoding_time_second: float
    encoding_iterations_cnt: int
    spatial_rate_bit: Optional[Tensor] = field(init=False)
    feature_rate_bpd: Optional[List[float]] = field(init=False, default_factory=lambda: [])
    img_size: Tuple[int, int] = field(init=False)
    n_pixels: int = field(init=False)
    detailed_rate_nn_bpd: DescriptorOverfitter = field(init=False)

    def __post_init__(self):
        for field in fields(self.loss_function_output):
            setattr(self, field.name, getattr(self.loss_function_output, field.name))
        self.img_size = self.original_frame.shape[-2:]
        self.n_pixels = self.original_frame.shape[-2] * self.original_frame.shape[-1] * 3
        self.detailed_rate_nn_bpd = {
            m_name: {wb: rate / self.n_pixels for wb, rate in mod.items()}
            for m_name, mod in self.detailed_rate_nn.items()
        }
        if 'detailed_rate_bit' in self.encoder_output.additional_data:
            detailed_rate_bit = self.encoder_output.additional_data.get('detailed_rate_bit')
            self.feature_rate_bpd = [x.sum(dim=(-1, -2, -3)) / (self.img_size[0] * self.img_size[1]) for x in detailed_rate_bit]

    def pretty_string(self, show_col_name: bool = False, mode: Literal['all', 'short'] = 'all', additional_data: Dict[str, Any] = {}) -> str:
        col_name, values = '', ''
        COL_WIDTH = 10
        for k in fields(self):
            if not self.should_be_printed(k.name, mode=mode): continue
            val = copy.deepcopy(getattr(self, k.name))
            if k.name == 'detailed_rate_nn_bpd':
                for s_name, s_rate in val.items():
                    col_name += f'{s_name + "_rate_bpd":<{COL_WIDTH}} '
                    values += f'{sum(s_rate.values()):<{COL_WIDTH}.6f} '
            elif k.name not in ['feature_rate_bpd']:
                col_name += f'{self.format_column_name(k.name):<{COL_WIDTH}} '
                values += f'{self.format_value(val, k.name):<{COL_WIDTH}} '
        for k, v in additional_data.items():
            col_name += f'{k:<{COL_WIDTH}} '
            values += f'{v:<{COL_WIDTH}} '
        return (col_name + '\n' + values) if show_col_name else values

    def should_be_printed(self, attribute_name: str, mode: str) -> bool:
        ATTRIBUTES = {
            'loss': ['short', 'all'], 'total_rate_bpd': ['short', 'all'], 'rate_img_bpd': ['short', 'all'],
            'rate_latent_bpd': ['short', 'all'], 'rate_nn_bpd': ['short', 'all'], 'encoding_time_second': ['short', 'all'],
            'encoding_iterations_cnt': ['short', 'all'], 'detailed_rate_nn_bpd': ['all'], 'n_pixels': ['all'], 'img_size': ['all']
        }
        return mode in ATTRIBUTES.get(attribute_name, [])

    def format_value(self, value, attribute_name=''):
        if attribute_name == 'img_size': return 'x'.join([str(t) for t in value])
        if isinstance(value, float): return f'{value:.6f}'
        if isinstance(value, Tensor): return f'{value.item():.6f}'
        return str(value)

    def format_column_name(self, col_name):
        LONG_TO_SHORT = {'rate_latent_bpd': 'lat_bpd', 'rate_img_bpd': 'img_bpd', 'rate_nn_bpd': 'nn_bpd', 'encoding_time_second': 'time_s', 'encoding_iterations_cnt': 'itr'}
        return LONG_TO_SHORT.get(col_name, col_name)


class FNLIC_VP(nn.Module):
    def __init__(self, encoder_param: OverfitterParameter, encoder_manager: EncoderManager, prefitter: Prefitter, img_t:Tensor):
        super().__init__()
        self.img_t_ori = img_t
        self.img_t = pad_image(img_t, encoder_param.n_latents-1)
        self.prefitter = prefitter
        self.prefitter.to_device('cpu')
        self.prefitter.eval()
        th.use_deterministic_algorithms(True)
        self.prior = prefitter.get_prior(self.img_t)
        th.use_deterministic_algorithms(False)
        self.encoder_param = encoder_param
        self.encoder = OverFitter(encoder_param)
        self.encoder_manager = encoder_manager
        # [已移除] PyTorch Compile

    def set_to_train(self): self.encoder.train()
    def set_to_eval(self): self.prefitter.eval(); self.encoder.eval()
    
    def forward(self, use_ste_quant: bool = False) -> EncoderOutput:
        out = self.encoder(self.img_t, self.prior, use_ste_quant)
        return EncoderOutput(img_bpd=out['img_bpd'], latent_bpd=out['latent_bpd'], additional_data={})

    def loss_function(self, encoder_out: EncoderOutput, rate_mlp_bpd: float = 0.)->LossFunctionOutput:
        loss = encoder_out.img_bpd + rate_mlp_bpd + encoder_out.latent_bpd
        return LossFunctionOutput(loss=loss, rate_nn_bpd=rate_mlp_bpd, rate_latent_bpd=encoder_out.latent_bpd, rate_img_bpd=encoder_out.img_bpd)

    # ------------------ main modified：Partial Update Strategy ------------------
    def one_training_phase(self, trainer_phase: TrainerPhase, alternating_update: bool = False):
        
      
        NET_UPDATE_FREQ = 5 if alternating_update else 1
        
        start_time = time.time()
        initial_encoder_logs = self.test()
        encoder_logs_best = initial_encoder_logs
        this_phase_best_model = OrderedDict((k, v.detach().clone()) for k, v in self.state_dict().items())

        self.set_to_train()

      
        net_params = []
        if 'arm' in trainer_phase.optimized_module: net_params += [*self.encoder.arm.parameters()]
        if 'upsampling' in trainer_phase.optimized_module: net_params += [*self.encoder.upsampling.parameters()]
        if 'synthesis' in trainer_phase.optimized_module: net_params += [*self.encoder.synthesis.parameters()]
        if 'latent' in trainer_phase.optimized_module: net_params += [*self.encoder.latents.parameters()] 
        if 'all' in trainer_phase.optimized_module: net_params = [*self.parameters()]

       
        real_latent_params = [*self.encoder.latents.parameters()]

        real_net_params = [p for p in net_params if not any(p is lp for lp in real_latent_params)]

      
        optimizer = th.optim.Adam(net_params, lr=trainer_phase.lr)

        scheduler = False
        if trainer_phase.scheduling_period:
            scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=trainer_phase.max_itr / trainer_phase.freq_valid, eta_min=0, last_epoch=-1)

        def linear_schedule(initial, final, cur, max_i): return cur * (final - initial) / max_i + initial
        
        cur_tmp = linear_schedule(trainer_phase.start_temperature_softround, trainer_phase.end_temperature_softround, 0, trainer_phase.max_itr)
        kumaraswamy_param = linear_schedule(trainer_phase.start_kumaraswamy, trainer_phase.end_kumaraswamy, 0, trainer_phase.max_itr)
        
        self.encoder.noise_quantizer.soft_round_temperature = cur_tmp
        self.encoder.ste_quantizer.soft_round_temperature = cur_tmp
        self.encoder.noise_quantizer.kumaraswamy_param = kumaraswamy_param

        cnt_record = 0
        show_col_name = True

        mode_str = f"交替更新 (1:{NET_UPDATE_FREQ})" if alternating_update else "全量更新"
        print(f" [系統] 模式: 原版 + {mode_str} (Threshold=原版 -0.0005)")

        # ========== train loop ==========
        for cnt in range(trainer_phase.max_itr):
            # Patience check logic
            if cnt - cnt_record > trainer_phase.patience:
                if not scheduler: break
                else:
                    self.load_state_dict(this_phase_best_model)
                    current_lr = scheduler.get_last_lr()[0]
                    cnt_record = cnt

            optimizer.zero_grad() 

         
            if alternating_update:
                is_net_update = (cnt % NET_UPDATE_FREQ == 0)
                for p in real_net_params: p.requires_grad = is_net_update
            
            # Forward
            out_forward = self.forward(use_ste_quant=trainer_phase.ste)
            loss_function_output = self.loss_function(out_forward)
            total_loss = loss_function_output.loss

            # Backward
            total_loss.backward()
            
            # Update
            nn.utils.clip_grad_norm_(self.parameters(), 1e-1, norm_type=2.0, error_if_nonfinite=False)
            optimizer.step()
            
            self.encoder_manager.iterations_counter += 1

            # Log / Validation Logic
            if ((cnt + 1) % trainer_phase.freq_valid == 0) or (cnt + 1 == trainer_phase.max_itr):
                
               
                for p in real_net_params: p.requires_grad = True

                self.encoder_manager.total_training_time_sec += time.time() - start_time
                start_time = time.time()
                encoder_logs = self.test()
                
                flag_new_record = False
                if encoder_logs.loss < encoder_logs_best.loss:
                   
                    flag_new_record = (encoder_logs.loss - encoder_logs_best.loss) < -0.0005

                if flag_new_record:
                    for k, v in self.state_dict().items(): this_phase_best_model[k].copy_(v)
                    encoder_logs_best = encoder_logs
                    cnt_record = cnt
                    log_rec = 'NEW BEST'
                else: log_rec = ''

                additional_data = {
                    'STE': trainer_phase.ste, 'lr': f'{trainer_phase.lr if not scheduler else scheduler.get_last_lr()[0]:.6f}',
                    'patience': (trainer_phase.patience - cnt + cnt_record) // trainer_phase.freq_valid,
                    'record': log_rec
                }
                logging.info(encoder_logs.pretty_string(show_col_name=show_col_name, mode='short', additional_data=additional_data))
                show_col_name = False

                cur_tmp = linear_schedule(trainer_phase.start_temperature_softround, trainer_phase.end_temperature_softround, cnt, trainer_phase.max_itr)
                kumaraswamy_param = linear_schedule(trainer_phase.start_kumaraswamy, trainer_phase.end_kumaraswamy, cnt, trainer_phase.max_itr)
                self.encoder.noise_quantizer.soft_round_temperature = cur_tmp
                self.encoder.ste_quantizer.soft_round_temperature = cur_tmp
                self.encoder.noise_quantizer.kumaraswamy_param = kumaraswamy_param
                
                if scheduler: scheduler.step()
                self.set_to_train()

       
        for p in real_net_params: p.requires_grad = True
        
        self.load_state_dict(this_phase_best_model)
        if trainer_phase.quantize_model: self.quantize_model()
        return self.test()

    @th.no_grad()
    def quantize_model(self):
        start_time = time.time()
        self.set_to_eval()
        module_to_quantize = {m_name: getattr(self.encoder, m_name) for m_name in self.encoder.modules_to_send}
        best_q_step = {k: None for k in module_to_quantize}

        for module_name, module in module_to_quantize.items():
            best_loss = 1e6
            module.save_full_precision_param()
            all_q_step = module._POSSIBLE_Q_STEP
            for q_step_w, q_step_b in itertools.product(all_q_step, all_q_step):
                current_q_step: DescriptorNN = {'weight': q_step_w, 'bias': q_step_b}
                if not module.quantize(current_q_step): continue
                rate_per_module = module.measure_laplace_rate()
                total_rate = sum(rate_per_module.values())
                
                if module_name == 'arm':
                    if ARMINT: self.encoder = self.encoder.to_device('cpu')
                    module.set_quant(FIXED_POINT_FRACTIONAL_MULT)

                encoder_out = self.forward()
                loss_out = self.loss_function(encoder_out, total_rate/self.encoder.img_size)
                
                if loss_out.loss < best_loss:
                    best_loss = loss_out.loss
                    best_q_step[module_name] = current_q_step
            
            module.quantize(best_q_step[module_name])
        
        logging.info(f'\nQuantization time: {time.time() - start_time:.1f}s')
        self.encoder.arm.set_quant(FIXED_POINT_FRACTIONAL_MULT)

    def one_training_loop(self, device: POSSIBLE_DEVICE, frame_workdir: str, start_time: str=0., alpha_init:float=1.0):
        logging.info(f'Training loop {self.encoder_manager.loop_counter + 1} / {self.encoder_manager.n_loops}')
        self.to_device(device)
        self.warmup(device, alpha_init=alpha_init)
        self.to_device(device)

        for idx_phase in range(self.encoder_manager.phase_idx, len(self.encoder_manager.preset.all_phases)):
            logging.info(f'Phase {idx_phase}'
            self.one_training_phase(self.encoder_manager.preset.all_phases[idx_phase], alternating_update=True)
            self.encoder_manager.phase_idx += 1
            logging.info(f'End phase results: {self.test().pretty_string(mode="short")}')

        encoder_logs = self.test()
        with open(f'{frame_workdir}results_loop_{self.encoder_manager.loop_counter + 1}.tsv', 'w') as f_out:
            f_out.write(encoder_logs.pretty_string(show_col_name=True, mode='all') + '\n')

        if self.encoder_manager.record_beaten(encoder_logs.loss):
            logging.info(f'New Best Loss: {encoder_logs.loss.cpu().item() :.6f}')
            self.encoder_manager.set_best_loss(encoder_logs.loss.cpu().item())
            with open(f'{frame_workdir}results_best.tsv', 'w') as f_out:
                f_out.write(encoder_logs.pretty_string(show_col_name=True, mode='all') + '\n')

        self.encoder_manager.loop_counter += 1
        self.encoder_manager.warm_up_done = False
        self.encoder_manager.phase_idx = 0
        return TrainingExitCode.END
        
    def overfit(self, device:POSSIBLE_DEVICE, work_dir: str, alpha_init:float) -> TrainingExitCode:
        return self.one_training_loop(device, work_dir, time.time(), alpha_init=alpha_init)
        
    def save(self, path: str): self.encoder.save(path)

    def warmup(self, device: POSSIBLE_DEVICE, alpha_init:float=1.0):
        start_time = time.time()
        training_preset = self.encoder_manager.preset
        logging.info(f'Warmup... {training_preset.get_total_warmup_iterations()} iters')

        for idx_warmup_phase, warmup_phase in enumerate(training_preset.all_warmups):
            if idx_warmup_phase == 0:
                all_candidates = [{'model': OverFitter(self.encoder_param, alpha_init=alpha_init), 'metrics': None, 'id': idx} for idx in range(warmup_phase.candidates)]
            else: all_candidates = all_candidates[:warmup_phase.candidates]

            training_phase = TrainerPhase(lr=warmup_phase.lr, max_itr=warmup_phase.iterations, start_temperature_softround=0.3, end_temperature_softround=0.3, start_kumaraswamy=2.0, end_kumaraswamy=2.0)

            for idx_candidate, candidate in enumerate(all_candidates):
                self.encoder = candidate.get('model')
                self.encoder.to_device(device)
                self.one_training_phase(training_phase, alternating_update=False)
                
                metrics = self.test() 
                self.encoder.to_device('cpu')
                all_candidates[idx_candidate] = {'model': self.encoder, 'metrics': metrics, 'id': candidate.get('id')}

            all_candidates = sorted(all_candidates, key=lambda x: x.get('metrics').loss)
            
        self.encoder = all_candidates[0].get('model')
        logging.info(f'Warmup done in {time.time() - start_time:.2f}s. Winner ID: {all_candidates[0].get("id")}')

    @th.no_grad()
    def test(self, training=True) -> EncoderLogs:
        rate_mlp = 0.
        rate_per_module = self.encoder.get_network_rate()
        for _, module_rate in rate_per_module.items():
            for _, param_rate in module_rate.items(): rate_mlp += param_rate
        self.set_to_eval()
        encoder_out = self.forward(use_ste_quant=False)
        loss_fn_output = self.loss_function(encoder_out, rate_mlp_bpd=rate_mlp/self.encoder.img_size)
        encoder_logs = EncoderLogs(
            loss_function_output=loss_fn_output, encoder_output=encoder_out, original_frame=self.img_t_ori,
            detailed_rate_nn=rate_per_module, quantization_param_nn=self.encoder.get_network_quantization_step(),
            encoding_time_second=self.encoder_manager.total_training_time_sec, encoding_iterations_cnt=self.encoder_manager.iterations_counter,
        )
        if training: self.set_to_train()
        return encoder_logs
    
    def test_inference_time(self) -> float:
        self.set_to_eval()
        latent = self.encoder.get_quantized_latent()
        latent = th.cat([cur_latent.flatten() for cur_latent in latent])
        max_latent_v = int(th.ceil(latent.abs().max()).item())
        with Timer(str(self.img_t.device)) as t:
            prior = self.prefitter.get_prior(self.img_t)
            self.encoder.inference_for_decode(self.img_t, prior, max_latent_v)
        return t.result

    def to_device(self, device: POSSIBLE_DEVICE):
        self.encoder.to_device(device)
        self.img_t = self.img_t.to(device)
        self.prior = self.prior.to(device)
        self.prefitter.to_device(device)

def load_fnlic_vp(src: str, overfitter_param: OverfitterParameter, img_t:Tensor, prefitter:Prefitter) -> FNLIC_VP:
    encoder = OverFitter(overfitter_param)
    encoder.load(src)
    fnlic = FNLIC_VP(encoder_param=overfitter_param, encoder_manager=None, img_t=img_t, prefitter=prefitter)
    fnlic.encoder = encoder
    return fnlic

def set_logger(dst:str):
    level = getattr(logging, 'INFO', None)
    handler = logging.FileHandler(dst)
    handler.setFormatter(logging.Formatter(''))
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(level)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler): logger.removeHandler(handler)

# ===========================================================================
# 3.(Main)
# ===========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,help='Path to the input image.png (RGB444)', type=str)
    parser.add_argument('-f', '--fnlic', default='', help='Path to overfitted model', type=str)
    parser.add_argument('-o', '--output', type=str, default="", help='Output bitstream path' )
    parser.add_argument('--workdir', help='Path to the working directory', type=str, default='../workspace')
    parser.add_argument('--prefitter_ckpt', help='Path to the prefitter checkpoint', type=str, default='../weight/prefitter.pth')
    parser.add_argument('--remove_workdir', help='Set to remove the working directory', action='store_true')
    parser.add_argument('--model_config', help='Path to the model configuration file', type=str, default='config/model_cfg.yaml')
    parser.add_argument('--training_config', help='Path to the overfit configuration file', type=str, default='config/training_cfg.yaml')
    args = parser.parse_args()

    with open(args.model_config, 'r') as f: model_cfg = yaml.safe_load(f)
    with open(args.training_config, 'r') as f: training_cfg = yaml.safe_load(f)
    assert len(training_cfg['alpha_inits']) > 0, 'Please provide at least one alpha_init'

    th._C._jit_set_profiling_executor(False)
    th._C._jit_set_texpr_fuser_enabled(False)
    th._C._jit_set_profiling_mode(False)

    workdir = f'{args.workdir.rstrip("/")}/'
    if not os.path.exists(workdir): os.mkdir(workdir)
    shutil.copyfile(args.model_config, os.path.join(args.workdir, 'model_cfg.yaml'))
    shutil.copyfile(args.training_config, os.path.join(args.workdir, 'training_cfg.yaml'))
    with open(f'{workdir}param.txt', 'w') as f_out: f_out.write(str(sys.argv))

    layers_synthesis = [x for x in model_cfg['layers_synthesis'].split(',') if x != '']
    layers_arm = [int(x) for x in model_cfg['layers_arm'].split(',') if x != '']
    device = get_best_device()
    logging.info(f'{"Device":<20}: {device}')

    if device == 'cpu':
        n_cores = os.getenv('SLURM_JOB_CPUS_PER_NODE')
        if n_cores is None: n_cores = os.cpu_count()
        n_cores=int(n_cores)
        th.set_flush_denormal(True)
        th.set_num_interop_threads(n_cores)
        th.set_num_threads(n_cores)
        subprocess.call('export OMP_PROC_BIND=spread', shell=True)
        subprocess.call('export OMP_PLACES=threads', shell=True)
        subprocess.call('export OMP_SCHEDULE=static', shell=True)
        subprocess.call(f'export OMP_NUM_THREADS={n_cores}', shell=True)
        subprocess.call('export KMP_HW_SUBSET=1T', shell=True)

    need_to_overfit = False
    alpha_inits = []
    for alpha_init in training_cfg['alpha_inits']:
        if not os.path.exists(os.path.join(workdir, f'fnlic_{alpha_init}.pth')):
            need_to_overfit = True
            alpha_inits.append(alpha_init)
    
    img_t = to_tensor(Image.open(args.input)).unsqueeze(0)

    overfitter_parameter = OverfitterParameter(
                img_shape=img_t.shape[-2:], layers_synthesis=layers_synthesis, layers_arm=layers_arm,
                n_latents=model_cfg['n_latents'], upsampling_kernel_size=model_cfg['upsampling_kernel_size'],
                img_bitdepth=model_cfg['img_bitdepth'], latent_bitdepth=model_cfg['latent_bitdepth'], freq_precision=16)

    prefitter_parameter = PrefitterParameter(img_bitdepth=model_cfg['img_bitdepth'], prior_arm_width=model_cfg['prefitter_width'], prior_arm_depth=model_cfg['prefitter_depth'])
    prefitter = Prefitter(prefitter_parameter)
    if os.path.exists(args.prefitter_ckpt):
        try: prefitter.load_state_dict(th.load(args.prefitter_ckpt, map_location='cpu'), strict=False)
        except: print('Could not load prefitter'); exit(1)
    else: print('No prefitter found'); exit(1)

    if need_to_overfit and args.fnlic == '':
        for alpha_init in alpha_inits:
            logging_path = os.path.join(workdir, f'fnlic_{alpha_init}.log')
            set_logger(logging_path)
            logging.info(args)
            encoder_manager = EncoderManager(preset_name='fnlic', start_lr=training_cfg['start_lr'], n_loops=training_cfg['n_overfit_loops'], n_itr=training_cfg['n_itr'])
            
            
            encoder = FNLIC_VP(encoder_param=overfitter_parameter, encoder_manager=encoder_manager, img_t=img_t, prefitter=prefitter)
            
            encoder.overfit(device, workdir, alpha_init)
            encoder.save(os.path.join(workdir, f'fnlic_{alpha_init}.pth'))
    
    if args.output != "":
        if args.fnlic == "":
            encoder_paths = glob(os.path.join(workdir, 'fnlic_*.pth'))
            bpd_min = 1e9
            bpd_min_path = ''
            for path in encoder_paths:
                alpha_init = path.split('_')[-1].replace('.pth', '')
                encoder = load_fnlic_vp(os.path.join(workdir, f'fnlic_{alpha_init}.pth'), overfitter_parameter, img_t, prefitter)
                encoder.to_device(device)
                bitstream_path = path.replace('.pth', '.fnlic')
                bpd = fnlic_encode(encoder, bitstream_path, device=device)
                logging_path = os.path.join(workdir, f'fnlic_{alpha_init}.log')
                set_logger(logging_path)
                logging.info(f'BPD: {bpd}')
                if bpd < bpd_min: bpd_min = bpd; bpd_min_path = bitstream_path
            shutil.copyfile(bpd_min_path, args.output)
        else:
            encoder = load_fnlic_vp(args.fnlic, overfitter_parameter, img_t, prefitter)
            encoder.to_device(device)
            bpd = fnlic_encode(encoder, args.output, device=device)
    if args.remove_workdir: subprocess.call(f'rm -r {workdir}', shell=True)
