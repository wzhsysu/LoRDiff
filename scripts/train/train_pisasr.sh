# choose exactly the GPUs you want
export CUDA_VISIBLE_DEVICES=7
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

accelerate launch train_pisasr.py \
  --pretrained_model_path stabilityai/stable-diffusion-2-1-base \
  --pretrained_model_path_csd stabilityai/stable-diffusion-2-1-base \
  --dataset_txt_paths /data2/zhihua/dataset/LSDIR/output.txt \
  --highquality_dataset_txt_paths /data2/zhihua/dataset/LSDIR/output.txt \
  --dataset_test_folder /data2/zhihua/LLIE/dataset/LOL-v2/Real_captured/Test \
  --dataset_train_folder /data2/zhihua/LLIE/dataset/LOL-v2/Real_captured/Train\
  --data_mode 'syn'  \
  --learning_rate 1e-4 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps 500 \
  --resolution_ori 512 --resolution_tgt 512 \
  --seed 123 \
  --output_dir /data2/zhihua/LLIE/train_pisasr_msssim_lpips_exp0.5_g1000_100000_synEXPJPEGBLUR2real_pix32_sem32_ram_hist \
  --timesteps1 1 \
  --lambda_lpips 2.0 --lambda_l2 1.0 --lambda_cd 0.0 \
  --pix_steps 100000 \
  --lora_rank_unet_pix 32 --lora_rank_unet_sem 32 \
  --min_dm_step_ratio 0.02 --max_dm_step_ratio 0.5 \
  --null_text_ratio 0.5 \
  --align_method adain \
  --deg_file_path params.yml \
  --tracker_project_name PiSASR \
  --is_module False \
  --eval_freq 500 \
  #--resume_path /data2/zhihua/LLIE/train_pisasr_without_color_fix_msssim_lpips_40000_SR_sys1/checkpoints/step_14001 
