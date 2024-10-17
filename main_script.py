import full_pipe_seg as sm

# Parameters
device = 'cpu'
image_dir = 'input_images'
result_dir = 'output_images'
checkpoint_path = 'pretrained/checkpoint_u2net.pth'
bucket_name = "boto-bucket-p"
file_name = 'fashion2.jpg'

# Run the full pipeline
sm.process_and_infer(device, checkpoint_path, file_name, image_dir, result_dir, bucket_name)
