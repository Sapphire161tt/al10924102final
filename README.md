# AI圖像識別, 南華大學
 * 10924135 
 * 10924131 
 * 10924102 
# 目錄
* 準備資料
* 準備數據
* 匯入模組
* 執行結果
* 與原作者區別

# 準備資料
* 準備一個可以打開google colab 的賬號
* 請使用我提供的Aiimage.ipynb來進行作業
  * (請先全部啟動一次,生成必要文件,正常來講是會出錯是正常現象,在準備好其他資料後在獨自一個一個執行)

# 準備數據
* [https://drive.google.com/file/d/12nitFO9fylbwvdghK1ufBoM7wD6Io_KZ/view?usp=sharing](https://drive.google.com/file/d/1LRniqJYNkQJQxiETnP7oHYAt_JqeAPEr/view?usp=sharing)
* 這個是網路上找到的模型庫
* 注意由於google colab 的資源提供量不多所以模型的大小太大,不然google colab會出錯

# 匯入模組
* 請點擊連結下載zip,將模型放置到雲端的執行檔目錄下，並改名為dog.h5
# 結論
* Aiimage.ipynb為訓練模型的過程，若想要預測自己的檔案可使用dog.h5和model.json即可預測
# 與原作者區別
* 原作者是在KAGLE平台上透過貓與狗的特徵庫來訓練出辨識動物品種的模型以辨識動物品種，但是因為Colab的空間有限，所以無法將特徵庫的資料全部放入COLAB中進行訓練，因此我們移除了特徵庫中的部分資料並修改程式碼，將原先辨識貓與狗品種的模型改變為僅辨識小狗品種的模型。
