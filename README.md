# AI圖像識別, 南華大學
 * 10924135 
 * 10924131 
 * 10924102
   
# 目錄
* 準備資料
* 執行結果
* 結論
  
# 準備資料
* 準備一個可以打開google colab 的帳號
* 請使用我提供的stable_diffusion_videos.ipynb來進行作業

# Making Videos
* Note: For Apple M1 architecture, use ```torch.float32``` instead, as ```torch.float16``` is not available on MPS.
* Note: 可將prompts改成自己想要的動物或物品
```python
from stable_diffusion_videos import StableDiffusionWalkPipeline
import torch

pipeline = StableDiffusionWalkPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
).to("cuda")

video_path = pipeline.walk(
    prompts=['a cat', 'a dog'],
    seeds=[42, 1337],
    num_interpolation_steps=3,
    height=512,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
    width=512,   # use multiples of 64 if > 512. Multiples of 8 if < 512.
    output_dir='dreams',        # Where images/videos will be saved
    name='animals_test',        # Subdirectory of output_dir where images/videos will be saved
    guidance_scale=8.5,         # Higher adheres to prompt more, lower lets model take the wheel
    num_inference_steps=50,     # Number of diffusion steps per image generated. 50 is good default
)
```

# 結論
* stable_diffusion_videos.ipynb提供製作可將影片中的小物品經由AI轉換為想要的物品
* 結果在sample.mp4
* 原作者(nateraw/stable-diffusion-videos)
[![影片标题]
