import librosa
import soundfile as sf
from util.rmvpe import RMVPE
from io import BytesIO
import os
from typing import List, Optional, Tuple
import numpy as np
import torch
from time import time as ttime
import logging

logger = logging.getLogger(__name__)

def main():

    audio, sampling_rate = sf.read("./1_video-144p_(Vocals).wav")
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    audio_bak = audio.copy()
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
    model_path = "./weights/rmvpe.pt"
    thred = 0.03  # 0.01
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rmvpe = RMVPE(model_path, is_half=False, device=device)
    t0 = ttime()
    f0 = rmvpe.infer_from_audio(audio, thred=thred)
    t1 = ttime()
    logger.info("%s %.2f", f0.shape, t1 - t0)
    print(f0)
    tensor = np.array(f0)

    # Write the tensor to a file
    with open('omniman.txt', 'w') as file:
        for value in tensor:
            file.write(f"{value}\n")


# def main():
#     # exp_dir=r"E:\codes\py39\dataset\mi-test"
#     # n_p=16
#     # f = open("%s/log_extract_f0.log"%exp_dir, "w")
#     printt(sys.argv)
#     featureInput = FeatureInput()
#     paths = []
#     inp_root = "%s/1_16k_wavs" % (exp_dir)
#     opt_root1 = "%s/2a_f0" % (exp_dir)
#     opt_root2 = "%s/2b-f0nsf" % (exp_dir)

#     os.makedirs(opt_root1, exist_ok=True)
#     os.makedirs(opt_root2, exist_ok=True)
#     for name in sorted(list(os.listdir(inp_root))):
#         inp_path = "%s/%s" % (inp_root, name)
#         if "spec" in inp_path:
#             continue
#         opt_path1 = "%s/%s" % (opt_root1, name)
#         opt_path2 = "%s/%s" % (opt_root2, name)
#         paths.append([inp_path, opt_path1, opt_path2])
#     try:
#         featureInput.go(paths[i_part::n_part], "rmvpe")
#     except:
#         printt("f0_all_fail-%s" % (traceback.format_exc()))
#     # ps = []
#     # for i in range(n_p):
#     #     p = Process(
#     #         target=featureInput.go,
#     #         args=(
#     #             paths[i::n_p],
#     #             f0method,
#     #         ),
#     #     )
#     #     ps.append(p)
#     #     p.start()
#     # for i in range(n_p):
#     #     ps[i].join()

if __name__ == "__main__":
    main()
    