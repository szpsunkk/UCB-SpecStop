# Acquire a new hardware campaign

The default CPU commands only analyze archived observations. This guide runs
new requests on an edge GPU and a target GPU. Use an isolated experimental
service and the same release commit on both machines. The server has no built-in
authentication; bind it only to a trusted experiment network.

## Install and obtain models

Use Python 3.10 or 3.12. Install PyTorch compatible with the hardware first.
For x86 CUDA 12.1 the recorded target used PyTorch 2.4.1:

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r hardware/requirements_hw.txt
```

On Jetson, use the NVIDIA/JetPack-compatible PyTorch build already installed or
the appropriate vendor wheel. Do not run the desktop-wheel command on Jetson.
`hardware/requirements_hw.txt` deliberately does not select or replace PyTorch.
The portable CPU pins support Python >=3.10; original edge metadata are retained
in the archived manifests for comparison.

Download/cache `Qwen/Qwen2.5-0.5B` on the edge and
`Qwen/Qwen2.5-7B-Instruct` on the server. Set `DRAFT_MODEL` and `TARGET_MODEL` to
your own local model directories. Models are loaded offline during the campaign.
Record the downloaded weight revision for a new experiment; the old archive
does not have that independent weight-level record.

Before measurement, CPU-only runtime checks can be run without model weights:

```bash
python -m pip install pytest==8.3.5
python -m pytest hardware/validate_hw.py hardware/validate_closed_loop_extensions.py \
  hardware/validate_delay_proxy.py hardware/validate_target_precision.py \
  hardware/validate_graph_draft.py experiments/table4/test_capability.py -q
```

## Prepare all eight sessions on the edge

Set the address for the server in your current shell. No address is hard-coded
in the public release. `CLOUD_HOST` is a hostname or IP address, without a scheme
or port. The proxy is local to the edge.

```bash
export CLOUD_HOST=your-target-hostname
export DRAFT_MODEL=/path/to/Qwen2.5-0.5B
python scripts/prepare_hardware.py \
  --server "http://${CLOUD_HOST}:8000" \
  --proxy-server http://127.0.0.1:18082 \
  --out-dir outputs/new_hardware
```

This copies the archived calibration and evaluation prompts, preserving the
arm set, methods, seeds, stopping rule and statistical family. It substitutes
deployment URLs and computes fresh code checksums. It refuses to overwrite an
existing campaign. The resulting plan is for reproducing the archived
calibration-transfer protocol, not calibrating a different device from scratch.
For a new scientific study, collect new calibration independently and freeze its
selection before evaluating; do not tune it on these 32 known questions.

## Run one session

On the target server, start a fresh process for each session:

```bash
export TARGET_MODEL=/path/to/Qwen2.5-7B-Instruct
export USE_TF=0 USE_FLAX=0
python scripts/launch_target.py --model "$TARGET_MODEL" \
  --host 0.0.0.0 --port 8000 --device cuda --dtype float16 \
  --attention-implementation sdpa --output-head-precision float32 \
  --session-verifier parallel
```

The launcher selects math SDPA, disabling flash and memory-efficient SDPA as
in the archived target run. The target verifier retains the numerical guard
and native multi-EOS handling. Use this launcher, not an independently
configured generic inference server.

On the edge, run the delay proxy in a separate terminal:

```bash
python hardware/delay_proxy.py --listen-host 127.0.0.1 --listen-port 18082 \
  --upstream-host "$CLOUD_HOST" --upstream-port 8000 --one-way-delay-ms 75
```

Run the first edge session:

```bash
export USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python hardware/capability_runner.py \
  --plan outputs/new_hardware/protocol/capability-01-plan.json \
  --out-dir outputs/new_hardware/hardware_run/capability-01 \
  --draft-model "$DRAFT_MODEL"
python scripts/record_session.py --campaign outputs/new_hardware --session capability-01
```

It executes 72 requests, randomizes method order using the frozen seed, persists
learner statistics within the session, and checks all exact outputs against
target-only. A failure stops the run and keeps logs. The output directory must
be new. Do not delete failed measurements or count selected retries as additional
independent sessions. Decide a restart policy before collecting a new study.

Stop only these experiment processes after completion, restart the target and
edge processes, and repeat for `capability-02` through `capability-08`. Register
each completed session. This process restart is part of the replication design;
running all 32 questions in one shared process changes the statistical unit.

## Analyze the new campaign

After all eight sessions are registered:

```bash
python -m experiments.table4.analyze --data-dir outputs/new_hardware \
  --out-dir outputs/new_hardware_analysis
python scripts/build_table4.py --out-dir outputs/new_hardware_analysis
```

Do not run the archived `check_results.py` against a fresh hardware campaign:
its latencies are new measurements. Use the preserved outcome and completeness
checks, and compare methods within the new paired sessions. Keep calibration
cost and runtime setup cost separate from measured request cost as documented.
Hardware logs are ignored by Git because they can contain hostnames and paths.
