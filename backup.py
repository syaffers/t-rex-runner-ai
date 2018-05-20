import argparse
import datetime
import glob
import os
import sys
import numpy as np

parser = argparse.ArgumentParser(
    description="Back up training data captured from the keylogging agent.")
parser.add_argument("-d", type=str, dest="data_folder", default="data/",
                    help="Path to the captured data (source) folder. "
                         "Default: data/")
parser.add_argument("-b", type=str, dest="backup_folder", default="backup/",
                    help="Path to the backup (destination) folder. "
                         "Default: backup/")
parser.add_argument("--dry-run", action="store_true",
                    help="Do a dry run backup.")

args = parser.parse_args()

current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

latest_backup = max(glob.glob(os.path.join(args.backup_folder, "*")),
                    key=os.path.getctime)

backup = np.load(latest_backup)
bk_images = backup['images']
bk_targets = backup['targets']

data = None
images = None
targets = None

# Go through all training images and actions in order.
for filepath in sorted(glob.glob(os.path.join(args.data_folder, "*")),
                       key=os.path.getctime):
    data = np.load(filepath)

    if '_x' in filepath:
        if images is None:
            images = data
        else:
            images = np.vstack([images, data])
    else:
        if targets is None:
            targets = data
        else:
            targets = np.vstack([targets, data])

    if args.dry_run:
        print("os.remove({})".format(filepath))
    else:
        os.remove(filepath)

# Stop if there is no data found.
assert data is not None, "No data was found in data folder."

bk_images = np.vstack([bk_images, images])
bk_targets = np.vstack([bk_targets, targets])

new_backup_fn = current_time + "_backup.npz"

save_msg = ("np.savez_compressed(os.path.join({}, {}), images=bk_images, "
            "targets=bk_targets)")

if args.dry_run:
    print(save_msg.format(args.backup_folder, new_backup_fn))
else:
    np.savez_compressed(os.path.join(args.backup_folder, new_backup_fn),
                        images=bk_images, targets=bk_targets)
