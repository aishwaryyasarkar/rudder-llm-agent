"""Exercise config -> submit -> CPU/GPU launcher using stub cluster commands."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def test_all_launch_paths_forward_inference_options(self):
        for mode in ('cpu', 'gpu'):
            for model in ('sage', 'gat'):
                with self.subTest(mode=mode, model=model), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    bin_dir = root/'bin'; bin_dir.mkdir()
                    commands = {
                        'activate': ':\n',
                        'scontrol': 'echo node1\n',
                        'srun': 'echo "inet 10.249.0.1/24"\n',
                        'lscpu': 'echo "Core(s) per socket: 4"; echo "Socket(s): 1"\n',
                        'python': 'printf "%s\\n" "$@" > "$CAPTURE"\n',
                        'sbatch': 'while [[ "$1" != "cpu.sh" && "$1" != "gpu.sh" ]]; do shift; done\nbash "$@"\n',
                    }
                    for name, body in commands.items():
                        path = bin_dir/name; path.write_text('#!/bin/bash\n'+body); path.chmod(0o755)
                    settings = dict(MODE=mode, MODEL=model, HIT_RATE='false', FP='0.5', DELTA='25', ALPHAS='0.05',
                                    DATASET_NAME='ogbn-arxiv', NUM_NODES='1', NUM_TRAINERS='1',
                                    NUM_SAMPLER_PROCESSES='0', QUEUE='debug', LOGS_DIR=str(root/'logs'),
                                    DATA_DIR=str(root/'data'), PROJ_PATH=str(ROOT), PARTITION_DIR=str(root/'partitions'),
                                    PARTITION_METHOD='metis', PREFETCHER_INIT='degree', DECISION_MODEL='gemma',
                                    ENABLE_FINETUNE='false', BATCH_SIZE='4', BATCHSIZE_EXP='false',
                                    COLLECT_TRAINING_FOR_CLASSIFIER='false', RUN_MODE='infer',
                                    CHECKPOINT_PATH=str(root/'model weights.pt'), OUTPUT_DIR=str(root/'output shards'),
                                    SAVE_SCORES='true', BATCH_SIZE_EVAL='2', PREDICTION_THRESHOLD='0.7')
                    config = root/'config.sh'
                    config.write_text('\n'.join(f'{k}={shlex.quote(v)}' for k,v in settings.items()))
                    capture = root/'capture'
                    env = dict(os.environ, PATH=str(bin_dir)+os.pathsep+os.environ['PATH'],
                               CAPTURE=str(capture), SLURM_JOB_ID='123', SLURM_JOB_NODELIST='node1')
                    result = subprocess.run(['bash', 'set_params.sh', '--config', str(config)],
                                            cwd=ROOT/'slurm', env=env, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    args = shlex.split(capture.read_text().splitlines()[-1])
                    for flag, expected in [('--run_mode','infer'), ('--checkpoint_path',settings['CHECKPOINT_PATH']),
                                           ('--output_dir',settings['OUTPUT_DIR']), ('--save_scores','true'),
                                           ('--batch_size_eval','2'), ('--prediction_threshold','0.7')]:
                        self.assertEqual(args[args.index(flag)+1], expected)
                    if model == 'sage':
                        self.assertEqual(args[args.index('--ml_model_dir')+1], str(ROOT/'classifier_models/gemma/trained_model'))


if __name__ == '__main__':
    unittest.main()
