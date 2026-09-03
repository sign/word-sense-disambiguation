"""Single entrypoint for running anything in a container: training scripts
(torchrun across nodes) or plain commands (--cmd), on Slurm (default) or
local Docker (--executor docker). Swapping infrastructure means changing
this file only — callers (Makefile) never invoke srun/docker directly.
"""
import os

# Must be set before the nemo_run import. NFS-backed on the cluster,
# local otherwise (docker executor on a workstation).
NEMORUN_HOME = os.environ.get('NEMORUN_HOME') or (
    '/mnt/nfs-1/amit/.nemo_run/wsd' if os.path.isdir('/mnt/nfs-1')
    else os.path.expanduser('~/.nemo_run/wsd'))
os.environ['NEMORUN_HOME'] = NEMORUN_HOME

import nemo_run as run  # noqa: E402

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=1)
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--script", type=str, help="python file/module to torchrun across nodes")
    what.add_argument("--cmd", type=str, help="inline shell command (one task, no torchrun)")
    parser.add_argument("--executor", choices=["slurm", "docker"], default="slurm")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--image_name", type=str, help="So enroot doesn't unsquash over and over")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--has_r2", action="store_true")
    # Check `sinfo -N` and ask about reservations before pinning; pass --nodelist per session.
    parser.add_argument("--nodelist", type=str, default=None, help="nodes we are allowed to use")
    parser.add_argument("--time", type=str, default="0", help='slurm time limit; "0" = unlimited')
    args, unknown = parser.parse_known_args()

    if args.cmd:
        task = run.Script(inline=args.cmd, entrypoint='bash')
    else:
        # --script ending in .py runs as a file; anything else runs as `python -m <module>`
        task = run.Script(args.script, args=unknown + ["--nodes", str(args.nodes)], entrypoint='python',
                          m=not args.script.endswith('.py'))

    home = os.path.expanduser('~')
    wandb_key = ''
    if os.path.exists(f'{home}/.wandbkey'):
        with open(f'{home}/.wandbkey') as r:
            wandb_key = r.read().strip()
    hf_token = ''
    if os.path.exists(f'{home}/.cache/huggingface/token'):
        with open(f'{home}/.cache/huggingface/token') as r:
            hf_token = r.read().strip()

    extra_env = {}
    container_env = []
    if args.has_r2:
        extra_env['R2_ENV'] = 'prod'
        extra_env['MOUNT_R2_CACHE_MAX_SIZE'] = '5000G'
        extra_env['MOUNT_R2_CACHE_DIR'] = '/scratch/r2_cache'
        with open(f'{home}/.op_svc_token') as r:
            extra_env['OP_SERVICE_ACCOUNT_TOKEN'] = r.read().strip()
        container_env = [
            'R2_ENV',
            'OP_SERVICE_ACCOUNT_TOKEN',
            'MOUNT_R2_CACHE_MAX_SIZE',
            'MOUNT_R2_CACHE_DIR'
        ]

    extra_env.update({k: v for k, v in os.environ.items()
                      if k.startswith(('WSD_', 'WANDB_PROJECT', 'WORDNET_URL'))})

    env_vars = dict(
        WANDB_API_KEY=wandb_key,
        PYTORCH_ALLOC_CONF="expandable_segments:True",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        PYTHONUNBUFFERED="1",
        TMPDIR='/tmp/.tmp',
        TORCHINDUCTOR_CACHE_DIR='/tmp/.torchinductor',
        TRITON_CACHE_DIR='/tmp/.triton',
        MEGATRON_CONFIG_LOCK_DIR="/tmp/.cache/huggingface",
        WANDB_DIR="/tmp/wandb",
        PYTHONFAULTHANDLER="1",
        HF_HOME="/scratch/.cache/huggingface",
        HF_TOKEN=hf_token,
        # non-NVIDIA base images (e.g. python:slim) don't set these, and without them enroot's hook injects no GPU
        NVIDIA_VISIBLE_DEVICES='all',
        NVIDIA_DRIVER_CAPABILITIES='compute,utility',
        # NCCL_DEBUG="info",
        # bootstrap/OOB over routable ethernet; data plane stays on IB verbs.
        # Unset, NCCL prefers ib* and picks ibp24s0 (10.100.x.5), unroutable cross-node.
        NCCL_SOCKET_IFNAME="enp41s0np0",
        GLOO_SOCKET_IFNAME="enp41s0np0",
        **extra_env
    )

    # torchrun for training scripts; plain single task for --cmd jobs
    launcher = 'torchrun' if args.script else None
    ntasks_per_node = 8 if args.script else 1

    if args.executor == 'docker':
        env_vars['HF_HOME'] = '/tmp/.cache/huggingface'
        executor = run.DockerExecutor(
            container_image=args.image,
            num_gpus=-1,  # all GPUs on the machine
            launcher=launcher,
            ntasks_per_node=ntasks_per_node,
            ipc_mode='host',
            ulimits=['memlock=-1', 'stack=67108864'],
            volumes=[f'{home}:{home}', f'{os.getcwd()}:{os.getcwd()}'],
            env_vars=env_vars,
            job_dir=NEMORUN_HOME + '/experiments/',
        )
    else:
        executor = run.SlurmExecutor(
            # Most of these parameters are specific to slurm
            account="rylo",
            partition="defq",
            ntasks_per_node=ntasks_per_node, # torchrun launcher replaces this with 1
            gpus_per_node=8,
            cpus_per_task=224,
            nodes=args.nodes,
            tunnel=run.LocalTunnel(job_dir=NEMORUN_HOME + '/experiments/'),
            container_image=args.image,
            launcher=launcher,
            time=args.time,
            mem="0",
            env_vars=env_vars,
            srun_args=[
                "--container-remap-root",
                "--container-writable",
                "--cpu-bind=none",
                "--container-name=" + (args.image_name or 'wsd'),
            ],
            container_env=container_env,
            container_mounts=[
                f'{home}:{home}',
                '/mnt/nfs-1:/mnt/nfs-1',
                '/scratch:/scratch',
                '/dev/infiniband:/dev/infiniband',
                '/sys/class/infiniband:/sys/class/infiniband:ro',
                '/sys/devices:/sys/devices:ro',
                '/dev/fuse:/dev/fuse'
            ],
            additional_parameters={
                **({"nodelist": args.nodelist} if args.nodelist else {}),
                # login shell's 8MB RLIMIT_MEMLOCK propagates into the job (PropagateResourceLimits=ALL)
                # and 4-node IB init dies on ibv_create_cq ENOMEM; NONE -> slurmd's unlimited.
                "propagate": "NONE",
            },
            # trailing /* -> `find *` under the cwd; the shell glob skips top-level dotfiles (.git etc.)
            packager=run.PatternPackager(include_pattern=os.getcwd() + '/*', relative_path=os.getcwd())
        )

    run.run(task, executor=executor, detach=args.detach)
