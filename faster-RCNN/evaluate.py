import imp

import wandb
from get_map import get_pred
from pascalvoc import main
import pkg_resources as pkg
import os
RANK = int(os.getenv('RANK', -1))
try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package import not local dir
    if pkg.parse_version(wandb.__version__) >= pkg.parse_version('0.12.2') and RANK in [0, -1]:
        try:
            wandb_login_success = wandb.login(timeout=30)
        except wandb.errors.UsageError:  # known non-TTY terminal issue
            wandb_login_success = False
        if not wandb_login_success:
            wandb = None
except (ImportError, AssertionError):
    wandb = None

wandb.init(project="faster-RCNN")
# currentPath = os.path.dirname(os.path.abspath(__file__))
# weight_list = [file for file in os.listdir(os.path.join(currentPath, 'logs')) if 'ep0' in file]
# weight_list = sorted(weight_list, key= lambda x: int(x[2:5]))
# # 得到真实框文件
# # get_pred()
# for weight in weight_list:
#     # 得到测试集上的预测结果：
#     try:
#         os.system("rm -r map_out/detection-results/")
#     except:
#         pass
#     get_pred(map_mode=1, model_path=os.path.join('/user/sunsiqi/yt/models/faster-rcnn-pytorch/logs', weight))
    
#     try:
#         map = main()
#         with open("temp.txt", "a") as f:
#             f.write("{}\n".format(map))
#         wandb.log({"metrics/map_0.5": map})
#     except:
#         pass
# get_pred(map_mode=1, model_path='/user/sunsiqi/yt/models/faster-rcnn-pytorch/logs/ep030.pth')
# map = main()

with open("temp.txt", 'r') as f:
    x = f.read().strip().split()

for map in x:
    wandb.log({"metrics/map_0.5": float(map)})
