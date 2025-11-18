import os
def load_sequence(seq_path, left_dir='left', right_dir='right', labels_file='labels.txt'):
    """Load left/right images and labels for a sequence."""
    left_imgs = sorted([f for f in os.listdir(os.path.join(seq_path, left_dir, 'data'))])
    right_imgs = sorted([f for f in os.listdir(os.path.join(seq_path, right_dir, 'data'))])
    
    labels = []
    with open(os.path.join(seq_path, labels_file)) as f:
        labels = [line.strip() for line in f.readlines()]
    
    return left_imgs, right_imgs, labels