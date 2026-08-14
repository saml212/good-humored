#!/bin/bash
# GRPO smoke v15 (2026-08-10) — v14 config unchanged; installed pure-python flash_attn.bert_padding GH-STUB into venv-verl site-packages (verl left_right_2_no_padding in the trainer imports it UNCONDITIONALLY — data reshaping only, no CUDA kernels; sdpa still does attention; no dist-info so transformers FA2 detection stays False).
# was v14 — v13 + actor_rollout_ref.model.use_remove_padding=False. v13 got through LoRA sync (384 params loaded) AND the full rollout; died in _compute_old_log_prob: verl rmpad path hard-imports flash_attn.bert_padding (not installed, no nvcc). Padded sdpa forward instead.
# was v13 — same config; GH-FIX applied in verl utils/vllm/utils.py: hijack passes weights_mapper=None to from_lora_tensors (Qwen3MoE fused-qkv mapper collapsed q/k/v LoRA entries -> slice_lora_b dim-1 IndexError). vllm_rollout/utils.py debug dump reverted.
# was v12 — same config; added GH-DEBUG instrumentation inside vllm slice_lora_b to identify the dim-1 lora_b (v11 proved all 384 tensors arrive 2-D at add_lora; the 1-D tensor is created inside vllm).
# was v11 — v10 config unchanged; run with GH-DEBUG patch in verl vllm_rollout/utils.py logging LoRA tensor shapes (v10 died: add_lora slice_lora_b IndexError dim-1 tensor; hunting which tensor arrives 1-D).
# was v10 — v9 + actor.fsdp_config.model_dtype=bf16. v9 diagnosis: the recurring 59854MiB WorkerDict spike at weight sync is load_fsdp_model_to_gpu reloading the FP32 FSDP shard (verl default model_dtype=fp32 for training: 30.5B x 4B / 2 ranks = 57GB) concurrent with vllm 30.5GB weight resume -> 87GB > 80GB. bf16 shard = 30.5GB -> sync peak ~64GB fits.
# was v9 — v8 + rollout.load_format=safetensors. v8 diagnosis: with load_format=dummy the LoRA BASE sync must materialize the full 61GB base per rank (collect_lora_params non-layered branch; layered_summon+base_sync_done=False raises ValueError telling you to use safetensors) concurrent with vllm 30.5GB mapped weights -> structurally OOM. safetensors => vllm loads base itself, base_sync_done=True from start, all syncs are adapter-only.
# was v8 — v7 + rollout.layered_summon=True. v7 diagnosis: FSDP offload works (1.66GB after init); vllm engine up; crash was at step-1 LoRA BASE weight sync — get_per_tensor_param without layered_summon gathers the FULL 61GB base model per rank while vllm has 30.5GB weights mapped (sampler: WorkerDict 59.8GB + vllm 15GB = 75.8/80GB -> cumem OOM). layered_summon gathers per-layer.
# was v7 — v6 + VERL_LOGGING_LEVEL=DEBUG (surface log_gpu_memory_usage lines incl. "After offload") + background GPU sampler. Diagnosing: vllm cumem OOM at weight alloc => did the FSDP actor release its 63GB before server launch?
# was v6 — v5 + RESTORED expandable_segments (FSDP1 flat-param shard clone OOMs from fragmentation without it: 58GB alloc + 18.9GB reserved-unallocated; v4 proved init passes with it). custom-allreduce IPC failure it caused in v4 is guarded by disable_custom_all_reduce.
# was: v5 (2026-08-10) — changes vs v4 (the attempt that died at
# vllm EngineCore init with custom_all_reduce.cuh:455 "invalid argument"):
#
# 1. DROP PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#    Root cause of the v4 crash: expandable-segment allocations cannot be
#    exported via cudaIpcGetMemHandle, so vLLM custom-allreduce IPC
#    registration fails with "invalid argument". Proof P2P itself is fine:
#    GLM serve runs TP=2 custom-allreduce on GPUs 2,3 without this env var.
#    verl manages expandable segments itself per phase
#    (engine_workers.py:702 False before rollout sync, :746 True for training).
# 2. +engine_kwargs.vllm.disable_custom_all_reduce=True — belt & suspenders;
#    NCCL allreduce over NVLink is fine for TP=2.
# 3. rollout.max_model_len=16384 — was model default 262144; vLLM requires KV
#    room for one max-len seq (~24GB) > budget -> would fail profiling next.
#    Driver smoke measured ~950 response tokens per 10-round session.
# 4. rollout.gpu_memory_utilization 0.45 -> 0.50 (KV headroom; peak phase =
#    weight sync: 30.5GB FSDP shard + 30.5GB vllm weights, KV resumed after
#    actor re-offload per engine_workers.sync_rollout_weights).
# 5. rollout.enforce_eager=True — skip cudagraph capture (memory + startup).
# 6. trainer.logger=[console] — remove wandb dependency for the smoke.
set -x
cd /data/good-humored/repo
source /data/good-humored/venv-verl/bin/activate
export GH_SESSION_DUMP_DIR=/data/good-humored/runs/rl_e_sessions && mkdir -p $GH_SESSION_DUMP_DIR && export HF_HOME=/data/good-humored/hf-cache
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=/data/good-humored/repo
export HYDRA_FULL_ERROR=1
export VERL_LOGGING_LEVEL=DEBUG
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( while true; do echo "--- $(date -u +%H:%M:%S)"; nvidia-smi -i 0,1 --query-compute-apps=pid,used_memory --format=csv,noheader; nvidia-smi -i 0,1 --query-gpu=index,memory.used --format=csv,noheader; sleep 5; done > /data/good-humored/logs/gpu_sampler_rl_e.log 2>&1 ) &
SAMPLER_PID=$!
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/data/good-humored/data-v5/sessions_train.parquet \
    data.val_files=/data/good-humored/data-v5/sessions_val.parquet \
    data.prompt_key=prompt \
    data.train_batch_size=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=/data/good-humored/runs/rl_d/merged_200_true \
    actor_rollout_ref.model.lora_rank=128 \
    actor_rollout_ref.model.lora_alpha=256 \
    "actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj]" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.max_model_len=16384 actor_rollout_ref.rollout.max_num_seqs=256 actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
    actor_rollout_ref.rollout.agent.default_agent_loop=humor_session_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=/data/good-humored/repo/env/verl_bridge/agent_loop_smooth_taste.yaml \
    "trainer.logger=[console]" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=200 \
    trainer.val_before_train=false \
    trainer.test_freq=-1 \
    trainer.save_freq=25 trainer.default_local_dir=/data/good-humored/runs/rl_e
RC=$?
kill $SAMPLER_PID 2>/dev/null
echo "LAUNCH SCRIPT EXIT CODE: $RC"
