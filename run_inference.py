import argparse
import torch
import os
from torch.utils.data import DataLoader
from collections import OrderedDict
from model import DGCNN_partseg
from model import DGCNN_regression 
from data import ShapeNetPart, MOHOReg  
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='DGCNN PartSeg Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Pretrained model path')
    parser.add_argument('--dataset_path', type=str, default='/data/DGCNN_datatest', help='Path to dataset root')
    parser.add_argument('--num_points', type=int, default=4096, help='Number of points per sample')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--class_choice', type=str, default=None, help='Optional single class to run inference on')
    parser.add_argument('--k', type=int, default=40, help='Number of neighbors for DGCNN')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--emb_dims', type=int, default=1024, help='Embedding dimensions')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout ratio')
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Load dataset (your custom loader already set up to read txt)
    test_dataset = MOHOReg(num_points=args.num_points, partition='test', class_choice=args.class_choice)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Model: match num_classes and seg_classes with training params
    # For ShapeNetPart it's usually seg_num_all = 50 when class_choice=None
    num_classes = len(test_dataset.cat2id) if args.class_choice is None else 1
    seg_classes = test_dataset.seg_num_all

    model = DGCNN_regression(args, seg_classes).cuda()

    # Load pretrained weights
    # Load checkpoint with prefix fix
    checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        new_key = k.replace('module.', '')  # remove 'module.' prefix if present
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)

    # Move model to GPU
    model = model.cuda()
    model.eval()
    
    output_dir = "regression_output_0820"
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if len(batch) == 3:
                data, label, _ = batch
            elif len(batch) == 2:
                data, label = batch
            else:
                data = batch
                label = torch.zeros((data.size(0),), dtype=torch.long)  # dummy label if missing

            data, label = data.cuda(), label.cuda()
            # Create zero one-hot vector for category input to the model (same as training)
            label_one_hot = torch.zeros(label.size(0), num_classes).cuda()
            label_one_hot.scatter_(1, label.view(-1, 1), 1)

            # Forward pass: regression output [B, 1, N]
            seg_pred = model(data.permute(0, 2, 1), label_one_hot)
            pred_values = seg_pred.squeeze(1)  # [B, N]

            # Convert data and prediction to CPU numpy arrays for saving
            for b in range(data.size(0)):
                points = data[b].cpu().numpy()  # [N, features]
                preds = pred_values[b].cpu().numpy()  # [N]
                
                print("points.shape:", points.shape)
                print("preds.shape:", preds.shape)
                # Prepare array with XYZ + predicted label columns
                output_array = np.hstack((points[:, :3], preds.reshape(-1, 1)))  # [N, 4]

                # Create filename with batch and sample index
                filename = os.path.join(output_dir, f"pred_regression_{batch_idx * data.size(0) + b:04d}.txt")
                np.savetxt(filename, output_array, fmt=['%.6f', '%.6f', '%.6f', '%6f'], delimiter=' ')

                print(f"Processed batch {batch_idx}, saved {data.size(0)} prediction files in txt format under '{output_dir}'")
                
if __name__ == '__main__':
    main()
