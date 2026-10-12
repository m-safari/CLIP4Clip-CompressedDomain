from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import numpy as np
import random
import os
import mlflow
import mlflow.pytorch
from metrics import compute_metrics
import time
import argparse
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.modeling import CLIP4ClipCompressed, VISUAL_BRANCHES, VISUAL_POOLINGS
from modules.optimization import BertAdam

from util import get_logger
from dataloaders.data_dataloaders import DATALOADER_DICT

global logger

# This script is hardcoded to a single GPU and the MSRVTT dataset.
# All torch.distributed / DistributedDataParallel and multi-dataset
# selection logic from the original has been removed.
DATATYPE = "msrvtt"
MLFLOW_EXPERIMENT_NAME = "clip4clip-compressed-domain"
# Written to --output_dir when a training run starts; `--resume_model auto` reads it.
RUN_ID_FILE = "last_run_id.txt"
# model.clip parameters that belong to the text branch (see --freeze_text).
TEXT_BRANCH_PREFIXES = ("token_embedding.", "positional_embedding", "transformer.", "ln_final.", "text_projection")

def get_args(description='CLIP4Clip on Retrieval Task (single-GPU, MSRVTT)'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument('--train_csv', type=str, default='data/.train.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/.val.csv',
                        help='Evaluated after every epoch; the "best" checkpoint is chosen on it. Use a '
                             'validation CSV from dataloaders/make_msrvtt_val_split.py so the test set '
                             'stays untouched during training.')
    parser.add_argument('--test_csv', type=str, default=None,
                        help='Held-out test CSV (MSRVTT_JSFUSION_test.csv). Only evaluated with --do_eval.')
    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')

    parser.add_argument('--num_thread_reader', type=int, default=4, help='DataLoader worker processes')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=12,
                        help='Temporal budget: number of GOPs decoded per video. All three modalities '
                             'are sampled from these same GOPs.')
    parser.add_argument('--residual_per_gop', type=int, default=1, help='Residuals sampled from each decoded GOP')
    parser.add_argument('--mv_per_gop', type=int, default=1, help='Motion vectors sampled from each decoded GOP')
    parser.add_argument('--max_train_samples', type=int, default=0,
                        help='Training samples drawn per epoch (a different seeded random subset each '
                             'epoch). 0 = the whole training set. Size it so one epoch fits a session.')
    parser.add_argument('--epochs_per_session', type=int, default=0,
                        help='Stop after this many epochs in this process (0 = no limit). --epochs stays '
                             'the TOTAL for the whole run, so the LR schedule is identical across sessions.')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--init_model", default=None, type=str, required=False,
                        help="Initial model weights: either a local file path, or an mlflow artifact URI "
                             "e.g. runs:/<run_id>/pytorch_model.last (or .best). To fully resume training "
                             "from a checkpoint, pass this together with --resume_model from the same run.")
    parser.add_argument("--resume_model", default=None, type=str, required=False,
                        help="'auto', or the mlflow artifact URI of the OPTIMIZER checkpoint to resume from, "
                             "e.g. runs:/<run_id>/pytorch_opt.last. Restores optimizer, AMP scaler, epoch, "
                             "global step and best score, and continues logging into the same mlflow run. "
                             "The matching pytorch_model.last is loaded too unless --init_model is given. "
                             "'auto' resumes the last run started in --output_dir, or starts a new run if "
                             "there is none, so the same command works for every session.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")
    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--slice_framepos', type=int, default=2, choices=[0, 1, 2],
                        help="Which GOPs are kept when a video has more than --max_frames: "
                             "0: first; 1: last; 2: uniformly spread.")
    parser.add_argument('--warmup_proportion', default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--freeze_text', type=int, default=1, choices=[0, 1],
                        help="1: freeze the whole text branch (embeddings, transformer, ln_final, "
                             "text_projection). 0: train it (at lr * coef_lr).")

    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")
    parser.add_argument("--visual_branches", nargs="+", default=list(VISUAL_BRANCHES), choices=VISUAL_BRANCHES,
                        help="Visual branches to build; drop any for an ablation, e.g. --visual_branches residual mv")
    parser.add_argument("--visual_pooling", default="modality", choices=VISUAL_POOLINGS,
                        help="modality: every modality weighs the same in the video embedding. "
                             "frame: one mean over all frames of all modalities (original behaviour).")
    parser.add_argument("--precision", default="amp", choices=["amp", "fp16"],
                        help="amp: fp32 weights and optimizer state, autocast + GradScaler (recommended). "
                             "fp16: the original hard half-precision weights, kept for comparison.")
    parser.add_argument("--grad_checkpoint", action='store_true',
                        help="Recompute the visual encoders' activations during backward instead of storing "
                             "them: much less GPU memory, roughly 30%% slower. Lets the batch size grow.")

    args = parser.parse_args()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args


def set_seed_logger(args):
    global logger
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args


def init_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)
    return device


def load_state_dict_from(uri):
    if uri.startswith("runs:/") or uri.startswith("models:/"):
        return mlflow.pytorch.load_state_dict(uri, map_location='cpu')
    return torch.load(uri, map_location='cpu')


def init_model(args, device):
    model_state_dict = load_state_dict_from(args.init_model) if args.init_model else None

    model = CLIP4ClipCompressed.from_pretrained(state_dict=model_state_dict, task_config=args)
    if args.precision == "amp" or device.type == "cpu":
        # fp32 master weights. On GPU the forward pass still runs in half
        # precision through autocast; only the weights, the gradients applied
        # to them and the optimizer state are full precision.
        model.float()
    model.to(device)
    return model


def freeze_parameters(args, model):
    """
    --freeze_text 1 freezes every parameter of the text branch.
    --freeze_layer_num N additionally freezes transformer blocks 0..N-1 of the
    I-frame tower (and of the text tower, if that is not frozen already).
    logit_scale and the residual / motion-vector encoders are never frozen here.
    """
    assert -1 <= args.freeze_layer_num <= 12
    for name, param in model.clip.named_parameters():
        if args.freeze_text and name.startswith(TEXT_BRANCH_PREFIXES):
            param.requires_grad = False
            continue
        if args.freeze_layer_num > -1:
            if name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num < args.freeze_layer_num:
                    param.requires_grad = False

    def count(prefix, trainable):
        return sum(p.numel() for n, p in model.named_parameters()
                   if n.startswith(prefix) and p.requires_grad == trainable)

    text_prefixes = tuple("clip." + p for p in TEXT_BRANCH_PREFIXES)
    for label, prefix in (("text", text_prefixes), ("iframe", "clip.visual."),
                          ("residual", "residual_encoder."), ("mv", "mv_encoder.")):
        logger.info("  %-8s branch: %.1fM trainable, %.1fM frozen", label,
                    count(prefix, True) / 1e6, count(prefix, False) / 1e6)


def prep_optimizer(args, model, num_train_optimization_steps, coef_lr=1.):
    # Frozen parameters are left out entirely: no weight decay, no optimizer state.
    param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    # CLIP-initialised weights get the small learning rate (lr * coef_lr). That is
    # model.clip AND the residual encoder, which starts as a copy of CLIP's visual
    # tower. Only weights trained from scratch (the motion-vector encoder) get lr.
    def is_pretrained(n):
        return n.startswith("clip.") or n.startswith("residual_encoder.")

    def is_no_decay(n):
        return any(nd in n for nd in no_decay)

    weight_decay = 0.2
    groups = [
        {'params': [p for n, p in param_optimizer if is_pretrained(n) and not is_no_decay(n)],
         'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in param_optimizer if not is_pretrained(n) and not is_no_decay(n)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if is_pretrained(n) and is_no_decay(n)],
         'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in param_optimizer if not is_pretrained(n) and is_no_decay(n)],
         'weight_decay': 0.0},
    ]
    optimizer_grouped_parameters = [g for g in groups if len(g['params']) > 0]

    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)

    return optimizer, model


def save_model(args, model, optimizer=None, train_state=None, type_name="last"):
    # type_name ("last" or "best") IS the artifact path -- no epoch in the
    # path, so saving again just overwrites the previous file for that type
    # instead of accumulating a new one per epoch.
    #
    #   last: full-precision weights + optimizer/scaler/progress. Everything
    #         needed to continue training exactly where it stopped.
    #   best: weights only, stored in half precision (inference only; nothing
    #         resumes from it), to stay within the storage budget.
    model_artifact_path = "pytorch_model.{}".format(type_name)
    state_dict = model.state_dict()
    if type_name != "last":
        state_dict = {k: (v.half() if v.is_floating_point() else v) for k, v in state_dict.items()}
    mlflow.pytorch.log_state_dict(state_dict, artifact_path=model_artifact_path)

    run_id = mlflow.active_run().info.run_id
    output_model_file = "runs:/{}/{}".format(run_id, model_artifact_path)
    logger.info("Model (%s) saved to %s", type_name, output_model_file)

    if optimizer is not None:
        optimizer_artifact_path = "pytorch_opt.{}".format(type_name)
        mlflow.pytorch.log_state_dict(dict(train_state, optimizer_state_dict=optimizer.state_dict()),
                                      artifact_path=optimizer_artifact_path)
        logger.info("Optimizer (%s) saved to runs:/%s/%s", type_name, run_id, optimizer_artifact_path)
    return output_model_file


def optimizer_diagnostics(optimizer):
    """
    Fraction of Adam second-moment entries that are exactly zero.

    With half-precision weights the squared gradient underflows to 0, the Adam
    denominator collapses to epsilon, and every step is lr * m / 1e-6. A healthy
    run shows ~0% after the first few steps; a large value means the optimizer
    is not behaving like Adam.
    """
    zero, total = 0, 0
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state.get(p)
            if not state:
                continue
            zero += int((state['next_v'] == 0).sum())
            total += state['next_v'].numel()
    return zero / max(total, 1)


def log_precision_summary(model, use_amp):
    counts = {}
    for p in model.parameters():
        key = (str(p.dtype).replace("torch.", ""), "trainable" if p.requires_grad else "frozen")
        counts[key] = counts.get(key, 0) + p.numel()
    logger.info("Precision: autocast+GradScaler=%s; parameters by dtype: %s", use_amp,
                ", ".join("{} {} {:.1f}M".format(d, t, n / 1e6) for (d, t), n in sorted(counts.items())))


def train_epoch(epoch, args, model, train_dataloader, device, optimizer, scaler, global_step):
    global logger
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model.train()
    log_step = args.n_display
    use_amp = scaler.is_enabled()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    start_time = time.time()
    total_loss, num_batches = 0., 0
    window_loss, window_batches = 0., 0
    failed_videos, skipped_batches = 0, 0
    grad_norm = float("nan")

    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        # Videos that failed to decode have no frames and no embedding: drop them.
        valid = model.valid_video_rows(batch[4], batch[6], batch[8])
        if not bool(valid.all()):
            failed_videos += int((~valid).sum())
            if int(valid.sum()) < 2:
                skipped_batches += 1
                continue
            batch = tuple(t[valid] for t in batch)

        pairs_text, pairs_mask, pairs_segment, iframe, iframe_mask, residuals, residuals_mask, mv, mv_mask = batch

        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = model(pairs_text, pairs_segment, pairs_mask,
                         iframe, iframe_mask, residuals, residuals_mask, mv, mv_mask)

        if not torch.isfinite(loss):
            logger.warning("Non-finite loss at epoch %d step %d; batch skipped.", epoch + 1, step + 1)
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        total_loss += loss.item()
        window_loss += loss.item()
        num_batches += 1
        window_batches += 1

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, 1.0))
            # Read before zero_grad(): BertAdam.get_lr() only reports parameters that have a gradient.
            lrs = sorted(set(optimizer.get_lr()))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # https://github.com/openai/CLIP/issues/46
            torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

            global_step += 1
            if global_step % log_step == 0:
                v_zero = optimizer_diagnostics(optimizer)
                peak_gb = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.
                avg_loss = window_loss / max(window_batches, 1)
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, GradNorm: %.3f, AdamV==0: %.1f%%, "
                            "AmpScale: %g, PeakMem: %.2fGB, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.9f' % itm) for itm in lrs]),
                            avg_loss, grad_norm, 100 * v_zero, scaler.get_scale() if use_amp else 1.,
                            peak_gb,
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                mlflow.log_metrics({
                    "train_loss": avg_loss,
                    "lr_min": lrs[0] if lrs else 0.,
                    "lr_max": lrs[-1] if lrs else 0.,
                    "grad_norm": grad_norm if np.isfinite(grad_norm) else -1.,
                    "adam_v_zero_frac": v_zero,
                    "amp_scale": scaler.get_scale() if use_amp else 1.,
                    "gpu_peak_gb": peak_gb,
                    "logit_scale": model.clip.logit_scale.exp().item(),
                }, step=global_step)
                window_loss, window_batches = 0., 0
                start_time = time.time()

    if failed_videos or skipped_batches:
        logger.warning("Epoch %d: %d training videos failed to decode and were excluded; %d batches skipped.",
                       epoch + 1, failed_videos, skipped_batches)
    total_loss = total_loss / max(num_batches, 1)
    return total_loss, global_step, failed_videos


def _compute_sim_matrix(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list):
    sim_matrix = []
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids = b1
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2, b2 in enumerate(batch_list_v):
            video_mask, = b2
            visual_output = batch_visual_output_list[idx2]
            b1b2_logits = model.get_similarity_logits(sequence_output, visual_output, input_mask, video_mask)
            each_row.append(b1b2_logits.cpu().detach().numpy())
        sim_matrix.append(np.concatenate(tuple(each_row), axis=-1))
    return sim_matrix


def eval_epoch(args, model, test_dataloader, device, step=0, prefix="val"):
    """
    Text-to-video and video-to-text retrieval over test_dataloader (one caption
    per video). Metrics go to mlflow as <prefix>_* at `step`.
    Videos that failed to decode are excluded, together with their captions,
    and their count is reported.
    """
    model = model.to(device)
    model.eval()
    use_amp = args.precision == "amp" and device.type == "cuda"
    failed_videos = 0

    with torch.no_grad():
        batch_list_t, batch_list_v = [], []
        batch_sequence_output_list, batch_visual_output_list = [], []

        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)

            valid = model.valid_video_rows(batch[4], batch[6], batch[8])
            if not bool(valid.all()):
                failed_videos += int((~valid).sum())
                if not bool(valid.any()):
                    continue
                batch = tuple(t[valid] for t in batch)

            pairs_text, pairs_mask, pairs_segment, iframe, iframe_mask, residuals, residuals_mask, mv, mv_mask = batch

            with torch.autocast(device_type=device.type, enabled=use_amp):
                sequence_output, visual_output, video_mask = model.get_sequence_visual_output(
                    pairs_text, pairs_segment, pairs_mask,
                    iframe, iframe_mask, residuals, residuals_mask, mv, mv_mask)

            batch_sequence_output_list.append(sequence_output.float())
            batch_list_t.append((pairs_mask, pairs_segment))

            batch_visual_output_list.append(visual_output.float())
            batch_list_v.append((video_mask,))

            print("{}/{}\r".format(bid, len(test_dataloader)), end="")

        if not batch_list_v:
            raise RuntimeError("Evaluation ({}): no video could be decoded.".format(prefix))

        sim_matrix = _compute_sim_matrix(model, batch_list_t, batch_list_v,
                                          batch_sequence_output_list, batch_visual_output_list)
        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

    logger.info("[%s] sim matrix size: %d, %d", prefix, sim_matrix.shape[0], sim_matrix.shape[1])
    if failed_videos:
        logger.warning("[%s] %d videos failed to decode and were excluded from the metrics.", prefix, failed_videos)
    else:
        logger.info("[%s] failed videos: 0", prefix)
    tv_metrics = compute_metrics(sim_matrix)
    vt_metrics = compute_metrics(sim_matrix.T)
    logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

    logger.info("[%s] Text-to-Video:", prefix)
    logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("[%s] Video-to-Text:", prefix)
    logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))

    mlflow.log_metrics({
        prefix + "_t2v_R1": tv_metrics['R1'],
        prefix + "_t2v_R5": tv_metrics['R5'],
        prefix + "_t2v_R10": tv_metrics['R10'],
        prefix + "_t2v_MedianR": float(tv_metrics['MR']),
        prefix + "_t2v_MeanR": float(tv_metrics['MeanR']),
        prefix + "_v2t_R1": vt_metrics['R1'],
        prefix + "_v2t_R5": vt_metrics['R5'],
        prefix + "_v2t_R10": vt_metrics['R10'],
        prefix + "_v2t_MedianR": float(vt_metrics['MR']),
        prefix + "_v2t_MeanR": float(vt_metrics['MeanR']),
        prefix + "_num_candidates": float(sim_matrix.shape[0]),
        prefix + "_num_failed_videos": float(failed_videos),
    }, step=step)

    return tv_metrics['R1']


def resolve_resume(args):
    """
    Work out which mlflow run (if any) this process continues, and fill in
    args.resume_model / args.init_model accordingly. Returns the run id or None.
    Needs the mlflow tracking URI to be set already.
    """
    run_id_file = os.path.join(args.output_dir, RUN_ID_FILE)
    run_id = None

    if args.resume_model == "auto":
        args.resume_model = None
        if os.path.exists(run_id_file):
            with open(run_id_file) as f:
                run_id = f.read().strip()
            artifacts = {info.path for info in mlflow.artifacts.list_artifacts(run_id=run_id)}
            if not args.do_train:
                # Evaluation only: load the selected model of that run.
                if not args.init_model:
                    args.init_model = "runs:/{}/pytorch_model.best".format(run_id)
            elif "pytorch_opt.last" in artifacts and "pytorch_model.last" in artifacts:
                args.resume_model = "runs:/{}/pytorch_opt.last".format(run_id)
            else:
                logger.info("Run %s has no finished epoch yet: training restarts from epoch 1 in the same run.",
                            run_id)
        else:
            logger.info("--resume_model auto: no previous run in %s, starting a new one.", args.output_dir)
    elif args.resume_model and args.resume_model.startswith("runs:/"):
        run_id = args.resume_model.split("runs:/", 1)[1].split("/", 1)[0]

    if args.do_train and args.resume_model and not args.init_model:
        args.init_model = args.resume_model.replace("pytorch_opt.", "pytorch_model.")
    return run_id


def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    params_to_log = dict(vars(args))

    mlflow.set_tracking_uri("sqlite:///" + os.path.join(args.output_dir, "mlflow.db"))
    # artifact_location can only be set when an experiment is first created --
    # it can't be changed afterward. If this experiment already exists (same
    # output_dir reused), its artifact root is whatever it was before; this
    # only puts artifacts under output_dir for a fresh experiment/output_dir.
    artifact_root = "file:" + os.path.join(args.output_dir, "mlartifacts")
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME, artifact_location=artifact_root)
    else:
        logger.info("Experiment '%s' already exists, artifact_location=%s (fixed at creation, not changed here)",
                    MLFLOW_EXPERIMENT_NAME, experiment.artifact_location)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # A resumed session reopens the SAME mlflow run, so "last"/"best" keep
    # being overwritten in one place and every epoch's metrics form one
    # continuous history.
    resume_run_id = resolve_resume(args)

    if resume_run_id:
        mlflow.start_run(run_id=resume_run_id)
        logger.info("Continuing mlflow run %s", resume_run_id)
    else:
        mlflow.start_run(run_name=os.path.basename(os.path.normpath(args.output_dir)))
    run_id = mlflow.active_run().info.run_id

    if args.do_train:
        with open(os.path.join(args.output_dir, RUN_ID_FILE), "w") as f:
            f.write(run_id)

    try:
        mlflow.log_params(params_to_log)
    except mlflow.exceptions.MlflowException:
        # Re-logging into a resumed run raises if any param value differs from
        # the original run -- not fatal, just skip.
        logger.info("Some params differ from the original run; skipped re-logging params.")

    device = init_device()

    tokenizer = ClipTokenizer()
    model = init_model(args, device)
    if args.init_model:
        logger.info("Model weights loaded from %s", args.init_model)

    logger.info("***** Parameters *****")
    freeze_parameters(args, model)
    use_amp = args.precision == "amp" and device.type == "cuda"
    log_precision_summary(model, use_amp)

    ## ####################################
    # train and eval
    ## ####################################
    if args.do_train:
        val_dataloader, val_length = DATALOADER_DICT[DATATYPE]["val"](args, tokenizer)
        train_dataloader, train_length = DATALOADER_DICT[DATATYPE]["train"](args, tokenizer)
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)
        logger.info("  Batch size = %d", args.batch_size_val)

        # The LR schedule is defined over the WHOLE run (--epochs epochs), however
        # many sessions it is split into.
        steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        if steps_per_epoch < 1:
            raise ValueError("The training set is smaller than one optimizer step.")
        num_train_optimization_steps = steps_per_epoch * args.epochs

        optimizer, model = prep_optimizer(args, model, num_train_optimization_steps, coef_lr=args.coef_lr)
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

        logger.info("***** Running training *****")
        logger.info("  Num examples per epoch = %d", train_length)
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Steps per epoch = %d, total steps = %d", steps_per_epoch, num_train_optimization_steps)

        start_epoch = 0
        best_score, best_epoch = -1., -1
        if args.resume_model:
            checkpoint = load_state_dict_from(args.resume_model)
            saved_total = checkpoint.get('t_total')
            if saved_total is not None and saved_total != num_train_optimization_steps:
                raise ValueError(
                    "Cannot resume: this run was started with {} total optimizer steps ({} per epoch), but the "
                    "current arguments give {} ({} per epoch). Keep --epochs, --batch_size, "
                    "--gradient_accumulation_steps, --max_train_samples and the training CSV identical in "
                    "every session; use --epochs_per_session to limit how much one session trains.".format(
                        saved_total, checkpoint.get('steps_per_epoch'), num_train_optimization_steps,
                        steps_per_epoch))
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint.get('scaler_state_dict') and use_amp:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            try:
                selection = mlflow.artifacts.load_dict("runs:/{}/train_state.json".format(run_id))
                best_score, best_epoch = float(selection['best_score']), int(selection['best_epoch'])
            except Exception:
                logger.info("No model-selection history found; the next evaluated epoch becomes the best.")
            logger.info("Resumed from %s: %d epochs done, best val R@1 so far %.2f (epoch %d)",
                        args.resume_model, start_epoch, best_score, best_epoch + 1)

        if start_epoch >= args.epochs:
            logger.info("All %d epochs are already done; nothing to train.", args.epochs)

        best_output_model_file = "runs:/{}/pytorch_model.best".format(run_id) if best_epoch >= 0 else "None"
        session_epochs = 0
        for epoch in range(start_epoch, args.epochs):
            if args.epochs_per_session and session_epochs >= args.epochs_per_session:
                logger.info("Reached --epochs_per_session=%d; run the same command again to continue.",
                            args.epochs_per_session)
                break

            # Everything random in an epoch depends on (seed, epoch) only, so an
            # epoch is the same whether or not the run was interrupted before it.
            random.seed(args.seed + epoch)
            np.random.seed(args.seed + epoch)
            torch.manual_seed(args.seed + epoch)
            train_dataloader.sampler.set_epoch(epoch)

            # The x-axis of every metric: optimizer steps since the start of the run.
            global_step = epoch * steps_per_epoch
            tr_loss, _, failed_videos = train_epoch(epoch, args, model, train_dataloader, device, optimizer,
                                                    scaler, global_step)
            global_step = (epoch + 1) * steps_per_epoch
            logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
            mlflow.log_metrics({"train_epoch_loss": tr_loss, "train_num_failed_videos": float(failed_videos),
                                "epoch": float(epoch + 1)}, step=global_step)

            train_state = {
                'epoch': epoch,
                'global_step': global_step,
                'steps_per_epoch': steps_per_epoch,
                't_total': num_train_optimization_steps,
                'scaler_state_dict': scaler.state_dict() if use_amp else None,
                'loss': tr_loss,
            }
            save_model(args, model, optimizer, train_state, type_name="last")

            R1 = eval_epoch(args, model, val_dataloader, device, step=global_step, prefix="val")
            if R1 > best_score:
                best_score, best_epoch = R1, epoch
                best_output_model_file = save_model(args, model, type_name="best")
            mlflow.log_dict({"best_score": best_score, "best_epoch": best_epoch, "last_evaluated_epoch": epoch},
                            "train_state.json")
            logger.info("The best model is: {} (epoch {}), the val R1 is: {:.4f}".format(
                best_output_model_file, best_epoch + 1, best_score))
            session_epochs += 1

    elif args.do_eval:
        if args.test_csv:
            eval_dataloader, eval_length = DATALOADER_DICT[DATATYPE]["test"](args, tokenizer)
            prefix = "test"
        else:
            eval_dataloader, eval_length = DATALOADER_DICT[DATATYPE]["val"](args, tokenizer)
            prefix = "val_only"
        logger.info("***** Running %s *****", prefix)
        logger.info("  Num examples = %d", eval_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        if not args.init_model:
            logger.warning("No --init_model: evaluating the UNTRAINED model (CLIP init, random motion-vector tower).")
        eval_epoch(args, model, eval_dataloader, device, step=0, prefix=prefix)

    mlflow.end_run()

if __name__ == "__main__":
    main()
