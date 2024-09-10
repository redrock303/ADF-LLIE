Here are an illustration for how to run our inference code and get the enhanced images.

  </br>
  &ensp; step 1 : run the inference script in stage1/inference.py and get the stage-1 enhanced result (Please refer to the IAT for more training details of the stage-1).
   </br>
  &ensp; step 2 : specify a LDRM method (such snr), and run the stage-2 result
   </br>
  &ensp; step 3 : run 'evaluation_psnr_ssim.py' for quantitative evaluation
   </br>
</br>

short instruction for training  the stage-1/2 models

  </br>
  &ensp; step 1 : run the train script in stage1/train.py 
   </br>
  &ensp; step 2 : save the intermediate results of stage1 for train the stage-2 model for fast stage-2 training
   </br>
  &ensp; step 3 : train the stage-2 model using stage2 bash command "python3 -m torch.distributed.launch --nproc_per_node=$num_gpu$ --master_port=4605 train.py"
   </br>
</br>