import cv2
import io 
import os
import uuid
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from networks import U2NET

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from data.base_dataset import Normalize_image
from gg_img_search import upload_image_to_s3, search_title



def load_checkpoint_mgpu(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=False)
    return model

def load_model(checkpoint_path, device='cpu'):
    net = U2NET(in_ch=3, out_ch=4)
    net = load_checkpoint_mgpu(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()
    return net    

def process_image(image_path, result_dir, transform_rgb, net, device):
    img = Image.open(image_path).convert('RGB')
    img_size = img.size
    img = img.resize((768, 768), Image.BICUBIC)

    image_tensor = transform_rgb(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)
    print(image_tensor.size())
    output_tensor = net(image_tensor.to(device))
    
    output_tensor = F.log_softmax(output_tensor[0], dim=1)
    output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_arr = output_tensor.cpu().numpy()

    # Change mask, wb
    output_arr[output_arr != 0] = 255
    output_arr[output_arr == 0] = 0

    # Convert to OpenCV format
    output_arr = output_arr.astype('uint8')
    print(output_arr.shape)
    # Apply dilation to enlarge the mask
    kernel = np.ones((7, 7), np.uint8)  # Larger kernel size to increase mask size
    output_arr = cv2.dilate(output_arr, kernel, iterations=2)  # More iterations for more enlargement

    # Convert back to PIL image
    output_img = Image.fromarray(output_arr, mode='L')
    
    # Resize back to original size
    output_img = output_img.resize(img_size, Image.BICUBIC)
    
    # Save the output image
    output_filename = os.path.basename(image_path)[:-4] + '_generated_bw.png'
    output_img.save(os.path.join(result_dir, output_filename))

    return output_img

def process_image_with_mask(image_path, mask_path):
    # Bước 1: Đọc ảnh gốc và ảnh mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Kiểm tra kích thước và resize mask nếu cần thiết
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Chuyển mask sang định dạng uint8
    mask = mask.astype(np.uint8)

    # Bước 2: Tách vùng mask từ ảnh gốc
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Bước 3: Tính toán màu trung bình của vùng quần áo
    masked_image_gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    mean_color = np.mean(masked_image_gray[mask > 0])

    # Bước 4: Chọn nền dựa trên màu sắc
    if mean_color > 127:
        background_color = [0, 0, 0]  # Nền đen
    else:
        background_color = [255, 255, 255]  # Nền trắng

    # Tạo ảnh mới với nền đã chọn
    background = np.full_like(image, background_color)

    # Áp dụng vùng mask lên ảnh nền
    result = background.copy()
    result[mask > 0] = masked_image[mask > 0]

    # Bước 5: Chuyển từ BGR sang RGB vì OpenCV đọc ảnh dưới dạng BGR
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    cv2.imwrite('processed_images/result.jpg', result)
    return result_rgb

def visualize_results(image, title, inference_time):
    plt.imshow(image)  # Directly pass the numpy array
    plt.title(title + "\n" + "Inference time: {:.4f} seconds".format(inference_time))
    plt.axis('off')  # Hide axis
    plt.show()

def full_pipe(device, checkpoint_path, file_name, image_dir, result_dir, bucket_name):
    net = load_model(checkpoint_path, device)
    image_path = f'{image_dir}/{file_name}'

    transforms_list = [transforms.ToTensor(), Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)

    output_img = process_image(image_path, result_dir, transform_rgb, net, device)
    result = process_image_with_mask(image_path, os.path.join(result_dir, file_name[:-4] + '_generated_bw.png'))
    image_url = upload_image_to_s3('processed_images/result.jpg', bucket_name, image_name=f"{str(uuid.uuid4())}.png")

    title = search_title(image_url)
    
    return result, title

def process_and_infer(device, checkpoint_path, file_name, image_dir, result_dir, bucket_name):
    start_time = time.time()
    
    output_img, title = full_pipe(device, checkpoint_path, file_name, image_dir, result_dir, bucket_name)
    
    inference_time = time.time() - start_time

    visualize_results(output_img, title, inference_time)
